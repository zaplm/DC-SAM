# DC-SAM: In-Context Segment Anything in Images and Videos via Dual Consistency

This repository is the official implementation of the paper: DC-SAM: In-Context Segment Anything in Images and Videos via Dual Consistency.

**Authors**: Pengfei Zhu, Mengshi Qi, Huadong Ma, Lu Qi, Xiangtai Li, Ming-Hsuan Yang

## Table of Contents

* [News](#news)
* [Highlights](#highlights)
* [Getting Started](#getting-started)
* [Training & Evaluation](#training--evaluation)
* [License](#license)

## News

**[2025/02/08]** The code and dataset for Image-to-image / Image-to-video In-Context Learning is released. Enjoy it:)

## Highlights

* **Dual Consistency SAM (DC-SAM) for one-shot segmentation**: Fully explores positive and negative features of visual prompts, generating high-quality prompts tailored for one-shot segmentation tasks.
* **Query cyclic-consistent cross-attention mechanism**: Ensures precise focus on key areas, effectively filtering out confusing components and improving accuracy and specificity in one-shot segmentation.
* **New video in-context segmentation benchmark (IC-VOS)**: Introduces a manually collected benchmark from existing video datasets, providing a robust platform for evaluating state-of-the-art methods.
* **Extension to SAM2 with mask tube design**: Enhances prompt generation for video object segmentation, achieving strong performance on the proposed IC-VOS benchmark.

## Getting Started

**Step 1**: clone this repository:

```
git clone https://github.com/zaplm/DC-SAM.git && cd DC-SAM
```

**Step 2**: create a conda environment and install the dependencies:
```
conda create -n DCSAM python=3.10 -y
conda activate DCSAM
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
cd segment-anything-2 && pip install -e .
cd ../segment-anything && pip install -e .
cd .. && pip install -r requirements.txt
```

## Data Prepare

## Training & Evaluation

### Training

It is recommand to train DC-SAM using 4 GPUs:

Train on image:
```
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=6224 train_image.py --epochs 50/100 --benchmark coco/pascal --lr 1e-4/2e-4 --bsz 2 --nshot 1 \
   --num_query 25 --sam_version 1/2 --nworker 8 --fold 0/1/2/3 --backbone resnet50/vgg16 --logpath log_name
```

Train on video (pretrain or finetune of image-to-video):
```
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=6224 train_image.py --epochs 40/10 --benchmark coco_mask_tube --lr 1e-4/1e-5 --bsz 8/1 --nshot 1 \
   --num_query 25 --data_type image/video --sam_version 2 --nworker 8 --backbone resnet50 --logpath log_name
```


### Validation

To evaluate on the IC-VOS:
```
python eval_video.py --coco_path /path/to/coco --icvos_path /path/to/icvos --ckpt /path/to/ckpt
```

To evaluate on few-shot segmentation benchmarks:
```
python eval_iamge.py --datapath /path/to/benchmark --benchmark pascal/coco --fold 0/1/2/3 --ckpt /path/to/ckpt
```

## License

This repository is licensed under [Apache 2.0](LICENSE).
