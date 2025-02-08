import os
import argparse
import numpy as np
import torch
from model.dc_sam2 import DC_SAM2
from common.metrics import db_eval_iou, db_eval_boundary
from data.ic_vos import DatasetIC_VOS
from common import utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings

# Set environment variable for Apple MPS fallback
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def db_statistics(per_frame_values):
    """Compute mean, recall and decay from per-frame evaluation.

    Arguments:
        per_frame_values (ndarray): per-frame evaluation

    Returns:
        tuple: mean, recall, decay
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(per_frame_values)
        recall = np.nanmean(per_frame_values > 0.5)

    n_bins = 4
    ids = np.round(np.linspace(1, len(per_frame_values), n_bins + 1) + 1e-10) - 1
    ids = ids.astype(np.uint8)

    d_bins = [per_frame_values[ids[i]:ids[i + 1] + 1] for i in range(0, 4)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        decay = np.nanmean(d_bins[0]) - np.nanmean(d_bins[3])

    return mean, recall, decay

def parse_arguments():
    parser = argparse.ArgumentParser(description='Visual Prompt Encoder Pytorch Implementation')
    parser.add_argument('--coco_path', type=str, default='/path/to/coco')
    parser.add_argument('--icvos_path', type=str, default='/path/to/ic-vos')
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--num_query', type=int, default=25)
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50', 'resnet101', 'swinb', 'dinov2b'])
    parser.add_argument('--data_type', type=str, default='video')
    parser.add_argument('--sam2_checkpoint', type=str, default='/path/to/sam2_checkpoint')
    parser.add_argument('--sam2_cfg', type=str, default='sam2.1_hiera_l.yaml')
    parser.add_argument('--lora', action='store_true', help='Use LoRA')
    parser.add_argument('--prior', action='store_true', help='Use Prior')
    parser.add_argument('--token_type', type=str, default='decoder', choices=['decoder', 'memory'])
    parser.add_argument('--add_token', action='store_true', help='Add token to the SAM2 model')
    parser.add_argument('--ckpt', type=str, default='/path/to/ckpt')
    return parser.parse_args()

def main():
    args = parse_arguments()
    utils.fix_randseed(0)

    # Select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    model_checkpoint = args.ckpt
    save_dict = os.path.join('vis_res', model_checkpoint.split('/')[-1].split('.')[0])
    os.makedirs(save_dict, exist_ok=True)

    model = DC_SAM2(args)
    state_dict = torch.load(model_checkpoint)
    new_state = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(new_state)
    model.to(device)

    dataset = DatasetIC_VOS(args.coco_path, args.icvos_path, 1)
    dataloader_test = DataLoader(dataset, batch_size=1, shuffle=False)

    count = 0
    j_buf = torch.zeros(len(dataset))
    f_buf = torch.zeros(len(dataset))
    pred_name = []

    with torch.inference_mode():
        for batch in tqdm(dataloader_test):
            batch = utils.to_cuda(batch)
            video_name, _, cls_id = batch['video_path'][0].split('/')[-3:]

            logit_mask, pred_mask = model(batch)

            np_pred = pred_mask[0].cpu().numpy()
            np_gt = batch['query_mask'][0].cpu().numpy()
            j_metric_res = db_eval_iou(np_pred, np_gt).mean() * 100
            f_metric_res = db_eval_boundary(np_pred, np_gt).mean() * 100
            j_buf[count] = j_metric_res
            f_buf[count] = f_metric_res
            print(f'J: {j_metric_res}, F: {f_metric_res}, length: {len(batch["frame_names"])}, class_id: {cls_id}')
            pred_name.append(f"{video_name}_{cls_id}")
            count += 1

    print(f'{model_checkpoint} Mean J: {j_buf.mean()}, Mean F: {f_buf.mean()}, Mean J&F: {(j_buf.mean() + f_buf.mean()) / 2}')

    print('Validation finished!')

if __name__ == "__main__":
    main()