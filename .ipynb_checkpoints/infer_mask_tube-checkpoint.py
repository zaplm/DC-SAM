import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import numpy as np
import pickle
import torch
from PIL import Image
import torch.nn.functional as F
from model.sam2 import SAM2
from common.metrics import db_eval_iou, db_eval_boundary
from new_dataset import DatasetNew
from common import utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import warnings
import cv2

def db_statistics(per_frame_values):
        """ Compute mean,recall and decay from per-frame evaluation.
        Arguments:
            per_frame_values (ndarray): per-frame evaluation

        Returns:
            M,O,D (float,float,float):
                return evaluation statistics: mean,recall,decay.
        """

        # strip off nan values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            M = np.nanmean(per_frame_values)
            O = np.nanmean(per_frame_values > 0.5)

        N_bins = 4
        ids = np.round(np.linspace(1, len(per_frame_values), N_bins + 1) + 1e-10) - 1
        ids = ids.astype(np.uint8)

        D_bins = [per_frame_values[ids[i]:ids[i + 1] + 1] for i in range(0, 4)]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            D = np.nanmean(D_bins[0]) - np.nanmean(D_bins[3])

        return M, O, D

parser = argparse.ArgumentParser(description='Visual Prompt Encoder Pytorch Implementation')
parser.add_argument('--nshot', type=int, default=1)
parser.add_argument('--num_query', type=int, default=25)
parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50', 'resnet101', 'swinb', 'dinov2b'])
parser.add_argument('--data_type', type=str, default='video')

# SAM2 config
parser.add_argument('--sam2_checkpoint', type=str, default='/root/autodl-tmp/weights/sam2.1_hiera_large.pt')
parser.add_argument('--sam2_cfg', type=str, default='sam2.1_hiera_l.yaml')
parser.add_argument('--lora', action='store_true', help='Use LoRA')
parser.add_argument('--token_type', type=str, default='decoder', choices=['decoder', 'memory'])
parser.add_argument('--add_token', action='store_true', help='Add token to the SAM2 model')
args = parser.parse_args()

utils.fix_randseed(0)

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

model_checkpoint = "/root/sawsam/logs/memory_token_p.pt"
save_dict = model_checkpoint.split('/')[-1].split('.')[0]
save_dict = os.path.join('vis_res', save_dict)
os.makedirs(save_dict, exist_ok=True)

model = SAM2(args)
state_dict = torch.load(model_checkpoint)
new_state = {}
for k, v in state_dict.items():
    name = k[7:]
    new_state[name] = v
model.load_state_dict(new_state)
model.to(device)

dataset = DatasetNew('/root/autodl-tmp', '/root/autodl-tmp/new_dataset', 1)
dataloader_test = DataLoader(dataset, batch_size=1, shuffle=False)

count = 0
ann_obj_id = 1
J_buf = torch.zeros(len(dataset))
F_buf = torch.zeros(len(dataset))
pred_name = []

with torch.inference_mode():
    for batch in tqdm(dataloader_test):
        batch = utils.to_cuda(batch)
        video_name, _,  cls_id = batch['video_path'][0].split('/')[-3:]
        
        logit_mask, pred_mask = model(batch)
        
        np_pred = pred_mask[0].cpu().numpy()
        # images = []
        # save_path = os.path.join(save_dict, video_name)
        # os.makedirs(save_path, exist_ok=True)
        # for idx, img_name in enumerate(batch['frame_names'][:512]):
        #     img_name = img_name[0]
        #     img = Image.open(os.path.join(os.path.dirname(os.path.dirname(batch['video_path'][0])), 'JPEGImages', img_name)).resize((np_pred.shape[1], np_pred.shape[2]))
        #     img = np.array(img)
        #     color_mask = np.zeros_like(img)
        #     single_mask = np.where(np_pred[idx], 255, 0).astype(np.uint8)
        #     color = (255, 0, 0)
        #     color_mask[single_mask == 255] = color
        #     color_img = cv2.addWeighted(img, 0.5, color_mask, 0.5, 0)
        #     color_img = Image.fromarray(color_img)
        #     color_img.save(f'{save_path}/{img_name}')
            # images.append(Image.fromarray(color_img))
        # images[0].save(f'{save_dict}/{video_name}_{cls_id}.gif', save_all=True, append_images=images[1:], duration=100, loop=0)
        np_gt = batch['query_mask'][0].cpu().numpy()
        j_metric_res = db_eval_iou(np_pred, np_gt).mean() * 100
        f_metric_res = db_eval_boundary(np_pred, np_gt).mean() * 100
        J_buf[count] = j_metric_res
        F_buf[count] = f_metric_res
        print(f'J: {j_metric_res}, F: {f_metric_res}, length: {len(batch["frame_names"])}, class_id: {cls_id}')
        pred_name.append(f"{video_name}_{cls_id}")
        count += 1

print(f'{model_checkpoint} Mean J: {J_buf.mean()}, Mean F: {F_buf.mean()}, Mean J&F: {(J_buf.mean() + F_buf.mean()) / 2}')
with open('memorytokenp.pkl', 'wb') as f:
    pickle.dump({'J_buf': J_buf, 'F_buf': F_buf, "pred_name": pred_name}, f)

print('val finished!')
    
    