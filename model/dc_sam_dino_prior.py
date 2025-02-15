r""" Dino Prior DC-SAM """
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_b
import model.base.resnet as models
import model.base.vgg as vgg_models
from torch.nn import BatchNorm2d as BatchNorm

from .base.transformer_decoder import transformer_decoder


def get_vgg16_layer(model):
    layer0_idx = range(0,7)
    layer1_idx = range(7,14)
    layer2_idx = range(14,24)
    layer3_idx = range(24,34)
    layer4_idx = range(34,43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]  
    layer0 = nn.Sequential(*layers_0) 
    layer1 = nn.Sequential(*layers_1) 
    layer2 = nn.Sequential(*layers_2) 
    layer3 = nn.Sequential(*layers_3) 
    layer4 = nn.Sequential(*layers_4)
    return layer0,layer1,layer2,layer3,layer4

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs, targets = inputs.flatten(1), targets.flatten(1)
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks

def sigmoid_focal_loss(inputs: torch.Tensor,
                       targets: torch.Tensor,
                       bs, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / bs


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class DC_SAM_DINO_Prior(nn.Module):
    def __init__(self, args, backbone, use_original_imgsize):
        super(DC_SAM_DINO_Prior, self).__init__()

        self.sam_version = args.sam_version
        self.nshot = args.nshot
        # 1. Backbone network initialization
        self.backbone_type = backbone
        self.use_original_imgsize = use_original_imgsize
        if backbone == 'vgg16':
            vgg_models.BatchNorm = BatchNorm
            vgg16 = vgg_models.vgg16_bn(pretrained=True)
            print(vgg16)
            self.layer0, self.layer1, self.layer2, \
                self.layer3, self.layer4 = get_vgg16_layer(vgg16)
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=True)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2, resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        elif backbone == 'resnet101':
            resnet = models.resnet101(pretrained=True)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2, resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        elif backbone == 'swinb':
            swin = swin_b()
            swin.load_state_dict(torch.load('/mapai/pfzhu/sam_utils/weights/swin_b.pth'))
            self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = \
                swin.features[0], swin.features[1], swin.features[2:4], swin.features[4:6], swin.features[6:]
        elif backbone == 'dinov2b':
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg', pretrained=False)
            self.dinov2.load_state_dict(torch.load('/mapai/pfzhu/sam_utils/weights/dinov2_vitb14.pth'))
            self.dinov2.requires_grad_(False)
        else:
            raise Exception('Unavailable backbone: %s' % backbone)
        if backbone != 'dinov2b':
            self.layer0.requires_grad_(False), self.layer1.requires_grad_(False), self.layer2.requires_grad_(False), self.layer3.requires_grad_(False), self.layer4.requires_grad_(False)
        hidden_dim = 256
        
        self.merge_1 = nn.Sequential(
            nn.Conv2d(hidden_dim*2+1, hidden_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.merge_2 = nn.Sequential(
            nn.Conv2d(hidden_dim*2+1, hidden_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        
        if self.sam_version == 2:
            self.sam_downscaling = nn.Sequential(
                nn.Conv2d(
                    hidden_dim // 8, hidden_dim // 4, kernel_size=2, stride=2
                ),
                LayerNorm2d(hidden_dim // 4),
                nn.GELU(),
                nn.Conv2d(
                    hidden_dim // 4, hidden_dim, kernel_size=2, stride=2
                ),
                nn.GELU(),
            )
        
        self.num_query = args.num_query
        
        self.transformer_decoder = transformer_decoder(args, args.num_query, hidden_dim, hidden_dim*2)
        self.neg_transformer_decoder = transformer_decoder(args, args.num_query, hidden_dim, hidden_dim*2)

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss()
    
    
    def forward(self, input, stage):
        if stage == 1:
            return self.stage1(*input)
        elif stage == 2:
            return self.stage2(*input)
        

    def stage1(self, query_img, support_img, support_mask, query_input, support_input):
        support_mask_ori = support_mask
        
        if self.sam_version == 2:
            c1, ln1, act1, c2, act2 = self.sam_downscaling
            feats_q0, feats_q1 = query_input['high_res_feats']
            query_sam = act1(ln1(c1(feats_q0) + feats_q1))
            query_sam = act2(c2(query_sam) + query_input['image_embed'])
            
            feats_s0, feats_s1 = support_input['high_res_feats']
            support_sam = act1(ln1(c1(feats_s0) + feats_s1))
            support_sam = act2(c2(support_sam) + support_input['image_embed'])
        else:
            query_sam = query_input
            support_sam = support_input

        with torch.no_grad():
            query_dino = F.interpolate(query_img, size=(896, 896), mode='bilinear', align_corners=True)
            support_dino = F.interpolate(support_img, size=(896, 896), mode='bilinear', align_corners=True)
            query_dino = self.dinov2.forward_features(query_dino)
            support_dino = self.dinov2.forward_features(support_dino)
            
            supp_feat_front = support_dino["x_prenorm"][:, :5]
            supp_feat_post = support_dino["x_prenorm"][:, 5:]
            support_mask = F.interpolate(support_mask_ori.unsqueeze(1).float(), size=(query_sam.size(2), query_sam.size(3)), mode='nearest')
            neg_support_mask = F.interpolate((1-support_mask_ori).unsqueeze(1).float(), size=(query_sam.size(2), query_sam.size(3)), mode='nearest')
            pos_supp_feat = torch.cat([supp_feat_front, supp_feat_post * support_mask.view(query_sam.size(0), -1, 1)], dim=1)
            neg_supp_feat = torch.cat([supp_feat_front, supp_feat_post * neg_support_mask.view(query_sam.size(0), -1, 1)], dim=1)
            
            query_feat_4 = query_dino["x_norm_patchtokens"].permute(0, 2, 1).view(query_sam.size(0), -1, query_sam.size(3), query_sam.size(3))
            supp_feat_4 = self.dinov2.norm(pos_supp_feat)[:, 5:].permute(0, 2, 1).view(query_sam.size(0), -1, query_sam.size(3), query_sam.size(3))
            neg_feat_4 = self.dinov2.norm(neg_supp_feat)[:, 5:].permute(0, 2, 1).view(query_sam.size(0), -1, query_sam.size(3), query_sam.size(3))
            
            pseudo_mask = self.get_pseudo_mask(supp_feat_4, query_feat_4, support_mask)
            neg_pseudo_mask = self.get_pseudo_mask(neg_feat_4, query_feat_4, neg_support_mask)

        prototype = self.mask_feature(support_sam, support_mask)
        supp_feat_bin = prototype.repeat(1, 1, query_sam.shape[2], query_sam.shape[3])
        neg_prototype = self.mask_feature(support_sam, neg_support_mask)
        neg_supp_feat_bin = neg_prototype.repeat(1, 1, query_sam.shape[2], query_sam.shape[3])
        supp_feat_1 = self.merge_1(torch.cat([support_sam, supp_feat_bin, support_mask*10], 1))                                                                                    
        query_feat_1 = self.merge_1(torch.cat([query_sam, supp_feat_bin, pseudo_mask*10], 1))
        neg_supp_feat_1 = self.merge_2(torch.cat([support_sam, neg_supp_feat_bin, neg_support_mask*10], 1))
        neg_query_feat_1 = self.merge_2(torch.cat([query_sam, neg_supp_feat_bin, neg_pseudo_mask*10], 1))

        protos = self.transformer_decoder((query_feat_1, supp_feat_1, support_mask, self.nshot), stage=1)
        neg_protos = self.neg_transformer_decoder((neg_query_feat_1, neg_supp_feat_1, neg_support_mask, self.nshot), stage=1)
        return (protos, neg_protos), prototype, query_feat_1, supp_feat_1
    
    def stage2(self,
               query_feat,
               pre_mask,
               neg_query_feat,
               output):
        pos_temp_query, neg_temp_query = output
        if pre_mask.dim() == 3:
            pre_mask = pre_mask.unsqueeze(1)
        pre_mask = F.interpolate(pre_mask.float(), size=(query_feat.size(2), query_feat.size(3)), mode='nearest')
        protos = self.transformer_decoder((query_feat, pre_mask, pos_temp_query.permute(1, 0, 2)), stage=2)
        neg_protos = self.neg_transformer_decoder((neg_query_feat, 1-pre_mask, neg_temp_query.permute(1, 0, 2)), stage=2)
        return (protos, neg_protos)

    def mask_feature(self, features, support_mask):
        mask = support_mask
        supp_feat = features * mask
        feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
        area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
        supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
        return supp_feat

    def predict_mask_nshot(self, args, batch, sam_model, nshot, input_point=None):

        # Perform multiple prediction given (nshot) number of different support sets
        logit_mask_agg = 0
        protos_set = []
        for s_idx in range(nshot):
            protos_sub, support_mask = self(args.condition, batch['query_img'], batch['support_imgs'][:, s_idx], batch['support_masks'][:, s_idx], False)
            protos_set.append(protos_sub)
        if nshot > 1:
            protos = torch.cat(protos_set, dim=1)
        else:
            protos = protos_sub

        low_masks, pred_mask = sam_model(batch['query_img'], batch['query_name'], protos,input_point)
        logit_mask = low_masks
        if self.use_original_imgsize:
            org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
            logit_mask = F.interpolate(logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)
        pred_mask = torch.sigmoid(logit_mask) >= 0.5

        pred_mask = pred_mask.float()
            
        logit_mask_agg += pred_mask.squeeze(1).clone()
        return logit_mask_agg, support_mask, logit_mask

    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        loss_bce = self.bce_with_logits_loss(logit_mask.squeeze(1), gt_mask.float())
        loss_dice = dice_loss(logit_mask, gt_mask, bsz)
        return loss_bce + loss_dice
        

    def train_mode(self):
        self.train()
        self.apply(fix_bn)

    def get_pseudo_mask(self, tmp_supp_feat, query_feat_4, mask):
        resize_size = tmp_supp_feat.size(2)
        tmp_mask = F.interpolate(mask, size=(resize_size, resize_size), mode='bilinear', align_corners=True)

        tmp_supp_feat_4 = tmp_supp_feat * tmp_mask
        q = query_feat_4
        s = tmp_supp_feat_4
        bsize, ch_sz, sp_sz, _ = q.size()[:]

        tmp_query = q
        tmp_query = tmp_query.reshape(bsize, ch_sz, -1)
        tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

        tmp_supp = s               
        tmp_supp = tmp_supp.reshape(bsize, ch_sz, -1) 
        tmp_supp = tmp_supp.permute(0, 2, 1)
        tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True) 

        cosine_eps = 1e-7
        similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)   
        similarity = similarity.max(1)[0].reshape(bsize, sp_sz*sp_sz)
        corr_query = similarity.reshape(bsize, 1, sp_sz, sp_sz)
        return corr_query