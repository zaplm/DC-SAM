import os
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np


class DatasetIC_VOS(Dataset):
    def __init__(self, cocopath, videopath, shot):
        self.cocopath = cocopath
        self.shot = shot
        videopath_list = os.listdir(videopath)
        self.videopath = []
        for vpath in videopath_list:
            vpath = os.path.join(videopath, vpath, 'Annotations')
            v_cls = os.listdir(vpath)
            self.videopath.extend([os.path.join(vpath, v) for v in v_cls])
        
        self.base_path = os.path.join(cocopath, 'COCO2014')
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        
        self.transform = transforms.Compose([transforms.Resize(size=(512, 512)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(img_mean, img_std)])
        
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.img_metadata = self.build_img_metadata()

    def __len__(self):
        return len(self.videopath)

    def __getitem__(self, idx):
        video_path = self.videopath[idx]
        cls_id = int(os.path.basename(video_path)) - 1
        support_imgs, support_masks, support_names = self.load_frame(cls_id)

        frame_names = [
            p for p in os.listdir(os.path.dirname(video_path).replace("Annotations", "JPEGImages"))
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        
        query_imgs = []
        query_masks = []
        for i in range(len(frame_names))[:512]:
            query_img = Image.open(os.path.join(os.path.dirname(video_path).replace("Annotations", "JPEGImages"),
                                                frame_names[i])).convert('RGB')
            org_qry_imsize = query_img.size
            query_mask = torch.tensor(np.array(Image.open(os.path.join(video_path, frame_names[i].replace(".jpg", ".png"))))).float()

            query_img = self.transform(query_img)
            query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()
            query_imgs.append(query_img)
            query_masks.append(query_mask)
        query_imgs = torch.stack(query_imgs)
        query_masks = torch.stack(query_masks)

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])
        for midx, smask in enumerate(support_masks):
            support_masks[midx] = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
        support_masks = torch.stack(support_masks)

        batch = {'query_img': query_imgs,
                 'query_mask': query_masks,

                 'org_query_imsize': org_qry_imsize,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,
                 'class_id': torch.tensor(cls_id),
                 'video_path': video_path,
                 'frame_names': frame_names}

        return batch

    def build_img_metadata_classwise(self):
        fold_n_subclsdata = os.path.join('data/splits/lists/coco_all/fss_list/val/sub_class_file_list_all.txt')
            
        with open(fold_n_subclsdata, 'r') as f:
            fold_n_subclsdata = f.read()
            
        sub_class_file_list = eval(fold_n_subclsdata)
        img_metadata_classwise = {}
        for sub_cls in sub_class_file_list.keys():
            img_metadata_classwise[sub_cls-1] = [data[0].split('/')[-1].split('.')[0] for data in sub_class_file_list[sub_cls]]
        return img_metadata_classwise

    def build_img_metadata(self):

        def read_metadata():
            fold_n_metadata = os.path.join('data/splits/lists/coco_all/fss_list/val/data_list_all.txt')
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
                
            fold_n_metadata = [data.split(' ')[0].split('/')[-1].split('.')[0] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        img_metadata = read_metadata()

        print('Total images are : %d' % (len(img_metadata)))

        return img_metadata

    def read_mask(self, name):
        mask_path = os.path.join(self.base_path, 'annotations')
        if 'val2014' in name:
            mask_path = os.path.join(mask_path, 'val2014')
        else:
            mask_path = os.path.join(mask_path, 'train2014')
        mask_path = os.path.join(mask_path, name)
        mask = torch.tensor(np.array(Image.open(mask_path + '.png')))
        return mask

    def load_frame(self, class_sample):
        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            support_names.append(support_name)
            if len(support_names) == self.shot: break

        support_imgs = []
        support_masks = []
        for support_name in support_names:
            if 'val2014' in support_name:
                base_path_s = os.path.join(self.base_path, 'val2014')
            else:
                base_path_s = os.path.join(self.base_path, 'train2014')
            support_imgs.append(Image.open(os.path.join(base_path_s, support_name + '.jpg')).convert('RGB'))
            support_mask = self.read_mask(support_name)
            support_mask[support_mask != class_sample + 1] = 0
            support_mask[support_mask == class_sample + 1] = 1
            support_masks.append(support_mask)

        return support_imgs, support_masks, support_names