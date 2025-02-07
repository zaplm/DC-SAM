import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from common.metrics import db_eval_iou, db_eval_boundary
import warnings
from tqdm import tqdm
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


# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

sam2_checkpoint = "/root/autodl-tmp/weights/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
# model_checkpoint = "/root/sawsam/logs/decoder_memory_l.pt"
# state_dict = torch.load(model_checkpoint)
# new_state = {}
# for k, v in state_dict.items():
#     name = k[7:]
#     if 'sam2' in name:
#         name = name[5:]
#         new_state[name] = v
# predictor.load_state_dict(new_state)
# predictor.to(device)

video_dir = "/root/autodl-tmp/new_dataset"
pred_dir = '/root/PFENet/vis_good'
save_dict = pred_dir.split('/')[-2]
save_dict = os.path.join('vis_res', save_dict)
os.makedirs(save_dict, exist_ok=True)

preds = os.listdir(pred_dir)

count = 0
J_buf = torch.zeros(len(preds))
F_buf = torch.zeros(len(preds))
pred_name = []
with torch.inference_mode():
    for pred in tqdm(preds):
        pred_path = os.path.join(pred_dir, pred)
        video_name, cls_id = pred.split('_')
        cls_id = int(cls_id.split('.')[0])
        frame_names = os.listdir(os.path.join(video_dir, video_name, 'JPEGImages'))
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        if len(frame_names) > 512:
            frame_names = frame_names[:512]
        pred_mask = Image.open(pred_path)
        # pred_mask = Image.open(os.path.join(video_dir, video_name, 'Annotations', str(cls_id), frame_names[0].replace(".jpg", ".png")))
        
        inference_state = predictor.init_state(video_path=os.path.join(video_dir, video_name, 'JPEGImages'))
        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
        
        _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            mask = np.array(pred_mask)
        )

        video_segments = []  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments.append((out_mask_logits[0] > 0.0).cpu().numpy())
        
        images = []
        save_path = os.path.join(save_dict, video_name)
        os.makedirs(save_path, exist_ok=True)

        for idx in range(len(frame_names)):
            img_name = frame_names[idx]
            img = Image.open(os.path.join(video_dir, video_name, 'JPEGImages', frame_names[idx])).resize((video_segments[idx][0].shape[1], video_segments[idx][0].shape[0]))
            img = np.array(img)
            color_mask = np.zeros_like(img)
            single_mask = np.where(video_segments[idx][0], 255, 0).astype(np.uint8)
            color = (255, 0, 0)
            color_mask[single_mask == 255] = color
            color_img = cv2.addWeighted(img, 0.5, color_mask, 0.5, 0)
            color_img = Image.fromarray(color_img)
            color_img.save(f'{save_path}/{img_name}')
            orig_img = Image.open(os.path.join(video_dir, video_name, 'JPEGImages', frame_names[idx]))
            anno_mask = Image.open(os.path.join(video_dir, video_name, 'Annotations', str(cls_id), frame_names[idx].replace(".jpg", ".png")))
            np_img = np.array(orig_img)
            np_mask = np.array(anno_mask)
            j_metrics_res = db_eval_iou(np_mask, video_segments[idx][0]) * 100
            f_metrics_res = db_eval_boundary(np_mask, video_segments[idx][0]) * 100
            J_buf[count] += j_metrics_res
            F_buf[count] += f_metrics_res
        J_buf[count] /= len(frame_names)
        F_buf[count] /= len(frame_names)
        pred_name.append(f"{video_name}_{cls_id}")
        print(f'J: {J_buf[count]}, F: {F_buf[count]}')
        count += 1
        
        
        predictor.reset_state(inference_state)

print(f'{pred_dir}: Mean J: {J_buf.mean()}, Mean F: {F_buf.mean()}, Mean J&F: {(J_buf.mean() + F_buf.mean()) / 2}')
# with open('infer_amnet_results.pkl', 'wb') as f:
#     pickle.dump({'J_buf': J_buf, 'F_buf': F_buf, "pred_name": pred_name}, f)