from model.dc_sam_prior import DC_SAM_Prior
from model.dc_sam import DC_SAM

import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2_video_predictor
from peft import LoraConfig, get_peft_model

class DC_SAM2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.data_type = args.data_type
        self.add_token = args.add_token
        self.token_type = args.token_type
        
        self.sam2 = build_sam2_video_predictor(args.sam2_cfg, ckpt_path=args.sam2_checkpoint, mode='train')
        if args.lora:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=['q_proj', 'v_proj', 'conv_s0', 'conv_s1'],
                lora_dropout=0.05,
                bias="none"
            )
            self.sam2 = get_peft_model(self.sam2, lora_config)
            self.sam2.print_trainable_parameters()
        else:
            self.sam2.requires_grad_(False)
            # self.sam2.memory_encoder.requires_grad_(True)
            # self.sam2.memory_attention.requires_grad_(True)
            
            self.sam2.sam_mask_decoder.requires_grad_(True)
            self.sam2.sam_mask_decoder.pred_obj_score_head.requires_grad_(False)
            self.sam2.sam_mask_decoder.iou_prediction_head.requires_grad_(False)
            if self.data_type == 'image':
                self.sam2.sam_mask_decoder.conv_s0.requires_grad_(False)
                self.sam2.sam_mask_decoder.conv_s1.requires_grad_(False)
        
        if args.prior:
            self.prompt_encoder = DC_SAM_Prior(args, args.backbone, False)
        else:
            self.prompt_encoder = DC_SAM(args, args.backbone, False)
        if self.data_type == 'video':
            state_dict = torch.load('/root/sawsam/logs/coco_all/best_model.pt')
            new_state = {}
            for k, v in state_dict.items():
                name = k[7:]
                new_state[name] = v
            self.load_state_dict(new_state)
    
    def forward(self, batch):
        inference_states = []
        bs = batch['query_img'].shape[0]
        if self.data_type == 'video':
            query_imgs = batch['query_img'][:, 0]
        else:
            query_imgs = batch['query_img']
        batch['query_img'] = batch['query_img'].view(-1, 3, 512, 512)
        
        batch['query_mask'] = batch['query_mask']
        state_imgs = F.interpolate(batch['query_img'], (1024,1024), mode='bilinear', align_corners=True).view(bs, -1, 3, 1024, 1024)
        query_imgs_state = F.interpolate(query_imgs, (1024,1024), mode='bilinear', align_corners=True)
        for i in range(bs):
            inference_state = self.sam2.init_state(frames=state_imgs[i], load_frames_from_path=False)
            inference_states.append(inference_state)
        query_state = self.sam2.init_state(frames=query_imgs_state, load_frames_from_path=False)
        query_sam = self.get_sam2_feature(query_imgs_state)    
        
        batch['support_imgs'] = batch['support_imgs'].reshape(-1, 3, 512, 512)
        supp_images = F.interpolate(batch['support_imgs'], (1024,1024), mode='bilinear', align_corners=True)
        
        batch['support_masks'] = batch['support_masks'].reshape(-1, 512, 512)
        supp_names = []
        for names in batch['support_names']:
            supp_names.extend(names)
        support_sam = self.get_sam2_feature(supp_images)
        
        protos, supp_prototype = self.prepare_prompt_inputs(query_state, query_imgs, batch['support_imgs'], batch['support_masks'], query_sam, support_sam)
        if self.add_token:
            for idx, inference_state in enumerate(inference_states):
                inference_state['other_token'] = supp_prototype[idx].squeeze()
                inference_state['token_type'] = self.token_type
        
        logit_masks, pred_masks = self.forward_tracking(protos, inference_states)

        return logit_masks, pred_masks
    
    def get_sam2_feature(self, infer_imgs):
        bs = infer_imgs.shape[0]
        _bb_feat_sizes = [
                (256, 256),
                (128, 128),
                (64, 64),
            ]
        
        with torch.no_grad():
            query_feats = self.sam2.image_encoder(infer_imgs)
            query_feats["backbone_fpn"][0] = self.sam2.sam_mask_decoder.conv_s0(
                query_feats["backbone_fpn"][0]
            )
            query_feats["backbone_fpn"][1] = self.sam2.sam_mask_decoder.conv_s1(
                query_feats["backbone_fpn"][1]
            )
        _, vision_feats, _, _ = self.sam2._prepare_backbone_features(query_feats)
        vision_feats[-1] = vision_feats[-1] + self.sam2.no_mem_embed
        feats = [
            feat.permute(1, 2, 0).reshape(bs, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], _bb_feat_sizes[::-1])
        ][::-1]
        query_feats_np = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        return query_feats_np
    
    def prepare_prompt_inputs(self, query_state, query_imgs, supp_imgs, supp_masks, query_sam, supp_sam):
        bs = query_imgs.shape[0]
        
        protos, supp_prototype = self.prompt_encoder(query_imgs, supp_imgs, supp_masks, query_sam, supp_sam, pre_mask)
        protos, supp_prototype, q_feat, s_feat = self.prompt_encoder((query_imgs, supp_imgs, supp_masks, query_sam, supp_sam), stage=1)
        
        pre_mask = []
        for i in range(bs):
            single_proto = (protos[0][i].unsqueeze(0), protos[1][i].unsqueeze(0))
            _, _, out_mask_logits = self.sam2.add_new_points_or_box(
                inference_state=query_state,
                frame_idx=i,
                obj_id=i,
                protos=single_proto
            )
            pre_mask.append((out_mask_logits[0] > 0.0).float())
        pre_mask = torch.stack(pre_mask, dim=0)
        
        protos = self.prompt_encoder((q_feat, pre_mask, s_feat, protos), stage=2)
        return protos, supp_prototype
    
    def forward_tracking(self, protos, inference_states):
        bs = protos[0].shape[0]
        logit_mask = [[] for _ in range(bs)]
        pred_mask = [[] for _ in range(bs)]
        
        for i in range(bs):
            inference_state = inference_states[i]
            single_proto = (protos[0][i].unsqueeze(0), protos[1][i].unsqueeze(0))
            _, _, out_mask_logits = self.sam2.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                protos=single_proto
            )
            
            for frame_idx, _, out_mask_logits in self.sam2.propagate_in_video(inference_state):
                logit_mask[i].append(out_mask_logits[0])
                pred_mask[i].append((out_mask_logits[0] > 0.0).float())
            logit_mask[i] = torch.cat(logit_mask[i], dim=0)
            pred_mask[i] = torch.cat(pred_mask[i], dim=0)
        logit_mask = torch.stack(logit_mask, dim=0)
        pred_mask = torch.stack(pred_mask, dim=0)
        
        return logit_mask, pred_mask