from segment_anything import sam_model_registry
from sam2.build_sam import build_sam2

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os


class SAM_plugin(nn.Module):
    def __init__(self, sam_version: int):
        super().__init__()
        self.vers = sam_version
        print('sam_version:', sam_version)
        
        if sam_version == 1:
            sam_model = sam_model_registry['vit_h']('/path/to/ckpt')
            self.image_encoder = sam_model.image_encoder
            self.prompt_encoder = sam_model.prompt_encoder
            self.mask_decoder = sam_model.mask_decoder
            self.image_encoder.requires_grad_(False)
            self.prompt_encoder.requires_grad_(False)
            self.mask_decoder.requires_grad_(False)
        elif sam_version == 2:
            sam2_model = build_sam2("sam2.1_hiera_l.yaml", '/path/to/ckpt')
            self.image_encoder = sam2_model.image_encoder
            self.prompt_encoder = sam2_model.sam_prompt_encoder
            self.mask_decoder = sam2_model.sam_mask_decoder
            self.no_mem_embed = sam2_model.no_mem_embed
            self.num_feature_levels = sam2_model.num_feature_levels
            self._bb_feat_sizes = [
                (256, 256),
                (128, 128),
                (64, 64),
            ]
            
        self.requires_grad_(False)
    
    
    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds, feat_sizes = 0, 0

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes
            

    def forward_img_encoder(self, query_img):
        query_img = F.interpolate(query_img, (1024,1024), mode='bilinear', align_corners=True)

        with torch.no_grad():
            query_feats = self.image_encoder(query_img)
            if self.vers == 2:
                query_feats["backbone_fpn"][0] = self.mask_decoder.conv_s0(
                    query_feats["backbone_fpn"][0]
                )
                query_feats["backbone_fpn"][1] = self.mask_decoder.conv_s1(
                    query_feats["backbone_fpn"][1]
                )
        return  query_feats
    
    def get_feat_sam(self, query_img, query_name):
        np_feat_path = '/path/to/save/feat/'
        if not os.path.exists(np_feat_path): os.makedirs(np_feat_path)
        query_feat_list = []
        for idx, name in enumerate(query_name):
            if '/root' in name:
                name = os.path.splitext(name.split('/')[-1])[0]
            try:
                sub_query_feat = torch.from_numpy(np.load(np_feat_path + name + '.npy')).to(query_img.device)
                query_feat_list.append(sub_query_feat)
            except:
                print('save feat for ', name)
                sub_query_feat = self.forward_img_encoder(query_img[idx, :, :, :].unsqueeze(0))
                query_feat_list.append(sub_query_feat)
                np.save(np_feat_path + name + '.npy', sub_query_feat.detach().cpu().numpy())
            del sub_query_feat
        query_feats_np = torch.cat(query_feat_list, dim=0)
        return query_feats_np
    
    def get_feat_sam2(self, query_img, query_name):
        # Since SAM2 feature is too big, we can't save it in numpy file
        query_feat_list = []
        for idx, name in enumerate(query_name):
            sub_query_feat = self.forward_img_encoder(query_img[idx, :, :, :].unsqueeze(0))
            query_feat_list.append(sub_query_feat)
            del sub_query_feat
        if self.vers == 1:
            query_feats_np = torch.cat(query_feat_list, dim=0)
        elif self.vers == 2:
            query_feats_np = {}
            for key in query_feat_list[0].keys():
                if key == 'vision_features':
                    query_feats_np[key] = torch.cat([x[key] for x in query_feat_list], dim=0)
                elif key == 'backbone_fpn':
                    data = [x[key] for x in query_feat_list]
                    query_feats_np[key] = [torch.cat([x[i] for x in data], dim=0) for i in range(len(data[0]))]
                    del data
            _, vision_feats, _, _ = self._prepare_backbone_features(query_feats_np)
            vision_feats[-1] = vision_feats[-1] + self.no_mem_embed
            feats = [
                feat.permute(1, 2, 0).reshape(query_img.shape[0], -1, *feat_size)
                for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
            ][::-1]
            query_feats_np = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        return query_feats_np

    def get_feat(self, query_img, query_name):
        if self.vers == 2:
            return self.get_feat_sam2(query_img, query_name)
        else:
            return self.get_feat_sam(query_img, query_name)

    def get_pormpt(self, protos, points_mask=None):
        if points_mask is not None :
            point_mask = points_mask

            postivate_pos = (point_mask.squeeze(0).nonzero().unsqueeze(0) + 0.5) * 64 -0.5
            postivate_pos = postivate_pos[:,:,[1,0]]
            point_label = torch.ones(postivate_pos.shape[0], postivate_pos.shape[1]).to(postivate_pos.device)
            point_prompt = (postivate_pos, point_label)
        else:
            point_prompt = None
        protos = protos
        return  protos, point_prompt

    def forward_prompt_encoder(self, points=None, boxes=None, protos=None, masks=None):
        q_sparse_em, q_dense_em = self.prompt_encoder(
                points=points,
                boxes=None,
                protos=protos,
                masks=None)
        return  q_sparse_em, q_dense_em
    
    def forward_mask_decoder(self, query_feats, q_sparse_em, q_dense_em, ori_size=(512,512)):
        if self.vers == 1:
            output = self.mask_decoder(
                image_embeddings=query_feats,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=q_sparse_em,
                dense_prompt_embeddings=q_dense_em,
                multimask_output=False)
        elif self.vers == 2:
            output = self.mask_decoder(
                    image_embeddings=query_feats['image_embed'],
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=q_sparse_em,
                    dense_prompt_embeddings=q_dense_em,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=query_feats['high_res_feats'])
        low_res_masks = output[0]
        low_masks = F.interpolate(low_res_masks, size=ori_size, mode='bilinear', align_corners=True)
            
        # from torch.nn.functional import threshold, normalize

        # binary_mask = normalize(threshold(low_masks, 0.0, 0))
        binary_mask = torch.where(low_masks > 0, 1, 0)
        return low_masks, binary_mask
    
    def forward(self, query_img, query_name, protos, points_mask=None):
        B,C, h, w = query_img.shape
        
        protos, point_prompt = self.get_pormpt(protos, points_mask)
        with torch.no_grad():

            query_feats = self.get_feat(query_img, query_name)

        q_sparse_em, q_dense_em = self.forward_prompt_encoder(
                points=point_prompt,
                boxes=None,
                protos=protos,
                masks=None)
            
        low_masks, binary_mask = self.forward_mask_decoder(query_feats, q_sparse_em, q_dense_em, ori_size=(h, w))

        return low_masks, binary_mask.squeeze(1)