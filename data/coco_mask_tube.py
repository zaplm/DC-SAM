import os
import random   
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
import cv2
import numbers
import collections


class Rotate(object):
    # Randomly rotate image & label with rotate factor in [rotate_min, rotate_max]
    def __init__(self, rotate, padding, ignore_label=255):
        assert (isinstance(rotate, collections.abc.Iterable) and len(rotate) == 2)
        if isinstance(rotate[0], numbers.Number) and isinstance(rotate[1], numbers.Number) and rotate[0] < rotate[1]:
            self.rotate = rotate
        else:
            raise (RuntimeError("segtransform.RandRotate() scale param error.\n"))
        assert padding is not None
        assert isinstance(padding, list) and len(padding) == 3
        if all(isinstance(i, numbers.Number) for i in padding):
            self.padding = padding
        else:
            raise (RuntimeError("padding in RandRotate() should be a number list\n"))
        assert isinstance(ignore_label, int)
        self.ignore_label = ignore_label

    def __call__(self, image, label):
        angle = self.rotate[0] + (self.rotate[1] - self.rotate[0]) * random.random()
        h, w = label.shape
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=self.padding)
        label = cv2.warpAffine(label, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return image, label


class HorizontalFlip(object):
    def __call__(self, image, label):
        image = cv2.flip(image, 1)
        label = cv2.flip(label, 1)
        return image, label


class VerticalFlip(object):
    def __call__(self, image, label):
        image = cv2.flip(image, 0)
        label = cv2.flip(label, 0)
        return image, label


class RandomGaussianBlur(object):
    def __init__(self, radius=5):
        self.radius = radius

    def __call__(self, image, label):
        image = cv2.GaussianBlur(image, (self.radius, self.radius), 0)
        return image, label


class DatasetCOCOMaskTube(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 80
        self.benchmark = 'coco'
        self.shot = shot
        self.split_coco = split if split == 'val2014' else 'train2014'
        self.base_path = os.path.join(datapath, 'COCO2014')
        self.transform = transform
        self.use_original_imgsize = use_original_imgsize

        self.class_ids = self.build_class_ids()
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.img_metadata = self.build_img_metadata()
        value_scale = 255
        mean = [0.485, 0.456, 0.406]
        mean = [item * value_scale for item in mean]
        self.mask_tube_transforms = [
            None,
            Rotate([-10, 10], padding=mean),
            HorizontalFlip(),
            VerticalFlip(),
            RandomGaussianBlur(radius=5)
        ]

    def __len__(self):
        return 10000 if self.split == 'trn' else 1000

    def __getitem__(self, idx):
        # ignores idx during training & testing and perform uniform sampling over object classes to form an episode
        # (due to the large size of the COCO dataset)
        query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize = self.load_frame()
        
        queries = []
        for trans in self.mask_tube_transforms:
            if trans is not None:
                query_img_temp, query_mask_temp = trans(np.array(query_img), query_mask.numpy())
            else:
                query_img_temp, query_mask_temp = np.array(query_img), query_mask.numpy()
            query_img_temp = Image.fromarray(query_img_temp)
            query_img_temp = self.transform(query_img_temp)
            query_mask_temp = torch.tensor(query_mask_temp)
            if not self.use_original_imgsize:
                query_mask_temp = F.interpolate(query_mask_temp.unsqueeze(0).unsqueeze(0).float(), query_img_temp.size()[-2:], mode='nearest').squeeze()
            queries.append((query_img_temp, query_mask_temp))
        random.shuffle(queries)
        query_img = [query[0] for query in queries]
        query_mask = [query[1] for query in queries]
        query_img = torch.stack(query_img)
        query_mask = torch.stack(query_mask)

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])
        for midx, smask in enumerate(support_masks):
            support_masks[midx] = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
        support_masks = torch.stack(support_masks)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,

                 'org_query_imsize': org_qry_imsize,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,
                 'class_id': torch.tensor(class_sample)}

        return batch

    def build_class_ids(self):
        # nclass_trn = self.nclass // self.nfolds
        class_ids_val = [x for x in range(self.nclass)]
        class_ids_trn = [x for x in range(self.nclass)]
        class_ids = class_ids_trn if self.split == 'trn' else class_ids_val

        return class_ids

    def build_img_metadata_classwise(self):
        if self.split == 'trn':
            split = 'train'
        else:
            split = 'val'
        fold_n_subclsdata = os.path.join('data/splits/lists/coco_all/fss_list/%s/sub_class_file_list_all.txt' % (split))
            
        with open(fold_n_subclsdata, 'r') as f:
            fold_n_subclsdata = f.read()
            
        sub_class_file_list = eval(fold_n_subclsdata)
        img_metadata_classwise = {}
        for sub_cls in sub_class_file_list.keys():
            img_metadata_classwise[sub_cls-1] = [data[0].split('/')[-1].split('.')[0] for data in sub_class_file_list[sub_cls]]
        return img_metadata_classwise

    def build_img_metadata(self):

        def read_metadata(split, fold_id):
            fold_n_metadata = os.path.join('data/splits/lists/coco_all/fss_list/%s/data_list_all.txt' % (split))
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
                
            fold_n_metadata = [data.split(' ')[0].split('/')[-1].split('.')[0] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        if self.split == 'trn':  # For training, read image-metadata of "the other" folds
            img_metadata += read_metadata('train', self.fold)
        elif self.split == 'val':  # For validation, read image-metadata of "current" fold
            img_metadata = read_metadata(self.split, self.fold)
        else:
            raise Exception('Undefined split %s: ' % self.split)

        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))

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

    def load_frame(self):
        class_sample = np.random.choice(self.class_ids, 1, replace=False)[0]
        query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        if 'val2014' in query_name:
            base_path = os.path.join(self.base_path, 'val2014')
        else:
            base_path = os.path.join(self.base_path, 'train2014')
        query_img = Image.open(os.path.join(base_path, query_name + '.jpg')).convert('RGB')
        query_mask = self.read_mask(query_name)

        org_qry_imsize = query_img.size

        query_mask[query_mask != class_sample + 1] = 0
        query_mask[query_mask == class_sample + 1] = 1

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
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

        return query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample.tolist(), org_qry_imsize


def get_mask_tube(img, mask, save_root):
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    mask_tube_transforms = [
            None,
            Rotate([-10, 10], padding=mean),
            HorizontalFlip(),
            VerticalFlip(),
            RandomGaussianBlur(radius=5)
        ]
    for idx, trans in enumerate(mask_tube_transforms):
        if trans is not None:
            query_img_temp, query_mask_temp = trans(np.array(img), mask)
        else:
            query_img_temp, query_mask_temp = np.array(img), mask
        query_img_temp = Image.fromarray(query_img_temp)
        query_mask_temp = Image.fromarray((query_mask_temp * 255).astype(np.uint8))
        query_img_temp.save(os.path.join(save_root, f'{idx}_img.jpg'))
        query_mask_temp.save(os.path.join(save_root, f'{idx}_mask.png'))