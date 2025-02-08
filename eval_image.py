import argparse

import torch

from model.dc_sam import DC_SAM
from model.dc_sam_prior import DC_SAM_Prior
from common.logger import AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
from SAM_plugin import SAM_plugin
from tqdm import tqdm


def evaluate(args, model: DC_SAM, sam_model: SAM_plugin, dataloader):

    utils.fix_randseed(args.seed)
    model.eval()
    average_meter = AverageMeter(dataloader.dataset)
    for batch in tqdm(dataloader):
        
        batch = utils.to_cuda(batch)
        query_sam = sam_model.get_feat(batch['query_img'], batch['query_name'])
        bs = batch['query_img'].size(0)
        batch['support_imgs'] = batch['support_imgs'].reshape(bs * args.nshot, 3, 512, 512)
        batch['support_masks'] = batch['support_masks'].reshape(bs * args.nshot, 512, 512)
        supp_names = []
        for names in batch['support_names']:
            supp_names.extend(names)
        support_sam = sam_model.get_feat(batch['support_imgs'], supp_names)
        
        protos, _, q_feat, s_feat = model((batch['query_img'], batch['support_imgs'], batch['support_masks'], query_sam, support_sam), stage=1)
        _, pre_mask = sam_model(batch['query_img'], batch['query_name'], protos)
        protos = model((q_feat, pre_mask, s_feat, protos), stage=2)
        
        low_masks, pred_mask = sam_model(batch['query_img'], batch['query_name'], protos)
        logit_mask = low_masks
        
        pred_mask = torch.sigmoid(logit_mask) > 0.5
        pred_mask = pred_mask.float()
        
        loss = model.compute_objective(logit_mask, batch['query_mask'])
        
        area_inter, area_union = Evaluator.classify_prediction(pred_mask.squeeze(1), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
    
    average_meter.write_result('Validation', 0, print_res=True)
        

if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Visual Prompt Encoder Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='/path/to/dataset')
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--sam_version', type=int, default=1)
    parser.add_argument('--bsz', type=int, default=2) # batch size = num_gpu * bsz default num_gpu = 4
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--nworker', type=int, default=12)
    parser.add_argument('--seed', type=int, default=321)
    parser.add_argument('--fold', type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument('--condition', type=str, default='mask', choices=['point', 'scribble', 'box', 'mask'])
    parser.add_argument('--use_ignore', type=bool, default=True, help='Boundaries are not considered during pascal training')
    parser.add_argument('--prior', action='store_true', help='Use Prior')
    parser.add_argument('--num_query', type=int, default=25)
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50', 'resnet101'])
    parser.add_argument('--ckpt', type=str, default='/path/to/ckpt')
    args = parser.parse_args()
    # Distributed setting
    device = torch.device('cuda', 0)
    Evaluator.initialize(args)
    
    utils.fix_randseed(args.seed)
    # Model initialization
    if args.prior:
        model = DC_SAM_Prior(args, args.backbone, False)
    else:
        model = DC_SAM(args, args.backbone, False)

    sam_model = SAM_plugin(args.sam_version)
    state_dict = torch.load(args.ckpt, map_location='cpu')
    new_state = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(new_state)
    sam_model.to(device)
    model.to(device)

    # Dataset initialization
    FSSDataset.initialize(img_size=512, datapath=args.datapath, use_original_imgsize=False)

    dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val', state='inference')
    # evaluate 
    evaluate(args, model, sam_model, dataloader_val)
