r""" Image-to-image In-context training code """
import argparse

import torch.optim as optim
import torch
import torch.distributed as dist

from model.dc_sam import DC_SAM
from model.dc_sam_prior import DC_SAM_Prior
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
from SAM_plugin import SAM_plugin


def train(args, epoch, model, sam_model, dataloader, optimizer, scheduler, training):
    r""" Train DC_SAM model """

    utils.fix_randseed(args.seed + epoch) if training else utils.fix_randseed(args.seed)
    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
        batch = utils.to_cuda(batch)
        query_sam = sam_model.get_feat_from_np(batch['query_img'], batch['query_name'])
        bs = batch['query_img'].size(0)
        batch['support_imgs'] = batch['support_imgs'].reshape(bs * args.nshot, 3, 512, 512)
        batch['support_masks'] = batch['support_masks'].reshape(bs * args.nshot, 512, 512)
        supp_names = []
        for names in batch['support_names']:
            supp_names.extend(names)
        support_sam = sam_model.get_feat_from_np(batch['support_imgs'], supp_names)
        
        protos, _, q_feat, s_feat = model((batch['query_img'], batch['support_imgs'], batch['support_masks'], query_sam, support_sam), stage=1)
        _, pre_mask = sam_model(batch['query_img'], batch['query_name'], protos)
        protos = model((q_feat, pre_mask, s_feat, protos), stage=2)

        low_masks, pred_mask = sam_model(batch['query_img'], batch['query_name'], protos)
        logit_mask = low_masks
        
        pred_mask = torch.sigmoid(logit_mask) > 0.5
        pred_mask = pred_mask.float()

        loss = model.module.compute_objective(logit_mask, batch['query_mask'])
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        area_inter, area_union = Evaluator.classify_prediction(pred_mask.squeeze(1), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Visual Prompt Encoder Pytorch Implementation')
    # Dataset
    parser.add_argument('--datapath', type=str, default='/path/to/dataset')
    parser.add_argument('--benchmark', type=str, default='coco', choices=['pascal', 'coco', 'coco_all'])
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--use_ignore', type=bool, default=True, help='Boundaries are not considered during pascal training')
    
    # Training
    parser.add_argument('--logpath', type=str, default='test')
    parser.add_argument('--bsz', type=int, default=2) # batch size = num_gpu * bsz default num_gpu = 4
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--nworker', type=int, default=8)
    parser.add_argument('--seed', type=int, default=321)
    
    # Model
    parser.add_argument('--sam_version', type=int, default=2)
    parser.add_argument('--num_query', type=int, default=25)
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50', 'resnet101', 'swinb', 'dinov2b'])
    parser.add_argument('--prior', action='store_true', help='Only use prior in the model')
    
    # Distributed setting
    parser.add_argument('--local-rank', type=int, default=-1, help='number of cpu threads to use during batch generation')
    args = parser.parse_args()
    # Distributed setting
    local_rank = args.local_rank
    dist.init_process_group(backend='nccl')
    print('local_rank: ', local_rank)
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    
    if utils.is_main_process():
        Logger.initialize(args, training=True)
    utils.fix_randseed(args.seed)
    # Model initialization
    if args.prior:
        model = DC_SAM_Prior(args, args.backbone, False)
    else:
        model = DC_SAM(args, args.backbone, False)
    if utils.is_main_process():
        Logger.log_params(model)

    sam_model = SAM_plugin(sam_version=args.sam_version)
    sam_model.to(device)
    model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # Device setup
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)
    # sam_model = torch.nn.parallel.DistributedDataParallel(sam_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)

    optimizer = optim.AdamW(model.module.parameters(),lr = args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    Evaluator.initialize(args)

    # Dataset initialization
    FSSDataset.initialize(img_size=512, datapath=args.datapath, use_original_imgsize=False)
    dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn', shot=args.nshot)

    dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val', shot=args.nshot)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= args.epochs * len(dataloader_trn))
    # Training 
    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    for epoch in range(args.epochs):

        trn_loss, trn_miou, trn_fb_iou = train(args, epoch, model, sam_model, dataloader_trn, optimizer, scheduler, training=True)
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou = train(args, epoch, model, sam_model, dataloader_val, optimizer, scheduler, training=False)

        # Save the best model
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            if utils.is_main_process():
                Logger.save_model_miou(model, epoch, val_miou)
        if utils.is_main_process():
            Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
            Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
            Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
            Logger.tbd_writer.flush()
    if utils.is_main_process():
        Logger.tbd_writer.close()
        Logger.info('==================== Finished Training ====================')