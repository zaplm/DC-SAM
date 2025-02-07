from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import numpy as np
import torch
import os


class DatasetDAVIS(Dataset):
    def __init__(self, data_root, split, img_size, version='2017', first_frame=False):
        self.data_root = data_root
        with open(os.path.join(data_root, 'DAVIS', 'ImageSets', version, split + '.txt'), 'r') as f:
            self.video_names = f.read().splitlines()
        self.video_paths = [os.path.join(self.data_root, 'DAVIS', 'JPEGImages', '480p', video) for video in self.video_names]
        
        self.img_paths = []
        for video_path in self.video_paths:
            imgs = os.listdir(video_path)
            imgs.sort()
            if first_frame:
                self.img_paths.extend([os.path.join(video_path, img) for img in imgs[1]])
            else:
                self.img_paths.extend([os.path.join(video_path, img) for img in imgs])
            
        self.img_mean = [0.485, 0.456, 0.406]
        self.img_std = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(self.img_mean, self.img_std)])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        ann_path = img_path.replace('JPEGImages', 'Annotations').replace('.jpg', '.png')
        query_img = self.transform(Image.open(self.img_paths[idx]))
        orig_mask = torch.from_numpy(np.array(Image.open(ann_path)))
        query_mask = torch.zeros_like(orig_mask)
        query_mask[orig_mask == 1] = 1
        query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()
        query_name = '_'.join(img_path.split('/')[-2:]).split('.')[0]
        
        org_qry_imsize = query_img.shape[-2:]
        
        supp_img_path = os.path.join(os.path.dirname(img_path), '00000.jpg')
        supp_ann_path = supp_img_path.replace('JPEGImages', 'Annotations').replace('.jpg', '.png')
        support_imgs = [self.transform(Image.open(supp_img_path))]
        support_masks = []
        support_names = ['_'.join(supp_img_path.split('/')[-2:]).split('.')[0]]
        origmask = torch.from_numpy(np.array(Image.open(supp_ann_path)))
        scmask = torch.zeros_like(origmask)
        scmask[origmask == 1] = 1 
        scmask = F.interpolate(scmask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()
        support_masks.append(scmask)
        support_imgs = torch.stack(support_imgs)
        support_masks = torch.stack(support_masks)
        
        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,

                 'org_query_imsize': org_qry_imsize,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,}
        return batch