r""" Image-to-video In-context training code """
import argparse

import torch.optim as optim
import torch
import torch.distributed as dist

from model.dc_sam2 import DC_SAM2
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset


def train(args, epoch, model, dataloader, optimizer, scheduler, training):

    utils.fix_randseed(args.seed + epoch) if training else utils.fix_randseed(args.seed)
    model.module.train() if training else model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)
    for idx, batch in enumerate(dataloader):
        batch = utils.to_cuda(batch)
        
        logit_mask, pred_mask = model(batch)

        loss = model.module.prompt_encoder.compute_objective(logit_mask, batch['query_mask'])
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Visual Prompt Encoder Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='/path/to/dataset')
    parser.add_argument('--benchmark', type=str, default='coco', choices=['pascal', 'coco', 'coco_all', 'coco_mask_tube'])
    parser.add_argument('--logpath', type=str, default='test')
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--sam_version', type=int, default=1)
    parser.add_argument('--data_type', type=str, default='image')
    parser.add_argument('--bsz', type=int, default=2) # batch size = num_gpu * bsz default num_gpu = 4
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--seed', type=int, default=321)
    parser.add_argument('--use_ignore', type=bool, default=True, help='Boundaries are not considered during pascal training')
    parser.add_argument('--local-rank', type=int, default=-1, help='number of cpu threads to use during batch generation')
    parser.add_argument('--num_query', type=int, default=25)
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50', 'resnet101', 'swinb', 'dinov2b'])
    
    # SAM2 config
    parser.add_argument('--sam2_checkpoint', type=str, default='/path/to/sam2_checkpoint')
    parser.add_argument('--sam2_cfg', type=str, default='sam2_hiera_l.yaml')
    parser.add_argument('--lora', action='store_true', help='Use LoRA')
    parser.add_argument('--token_type', type=str, default='decoder', choices=['decoder', 'memory'])
    parser.add_argument('--add_token', action='store_true', help='Add token to the SAM2 model')
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
    model = DC_SAM2(args)
    if utils.is_main_process():
        Logger.log_params(model)
        
    model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # Device setup
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)

    optimizer = optim.AdamW([
        {'params': model.module.prompt_encoder.parameters()},
        {'params': model.module.sam2.sam_mask_decoder.parameters(), "lr": args.lr},
        {'params': model.module.sam2.memory_encoder.parameters(), "lr": args.lr},
        {'params': model.module.sam2.memory_attention.parameters(), "lr": args.lr},
        ],lr = args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    Evaluator.initialize(args)

    # Dataset initialization
    FSSDataset.initialize(img_size=512, datapath=args.datapath, use_original_imgsize=False)
    dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, 0, 'trn', shot=args.nshot)

    dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, 0, 'val', shot=args.nshot)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= args.epochs * len(dataloader_trn))
    # Training 
    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        trn_loss, trn_miou, trn_fb_iou = train(args, epoch, model, dataloader_trn, optimizer, scheduler, training=True)
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou = train(args, epoch, model, dataloader_val, optimizer, scheduler, training=False)

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