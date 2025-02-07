import os
import os.path as osp
import ujson

root = '/apdcephfs/private_pfzhu/VRP-SAM/data/splits/lists/pascal/fss_list/val'
save_root = '/apdcephfs/private_pfzhu/VRP-SAM/data/splits/lists/coco2pascal/fss_list/val'
data_lists = [[] for _ in range(4)]
sub_class_lists = [{} for _ in range(4)]
categories = [
    (1, 0),  # aeroplane
    (2, 1),  # bicycle
    (3, 2),  # bird
    (4, 0),  # boat
    (5, 3),  # bottle
    (6, 1),  # bus
    (7, 2),  # car
    (8, 3),  # cat
    (9, 0),  # chair
    (10, 3), # cow
    (11, 0), # diningtable
    (12, 0), # dog
    (13, 1), # horse
    (14, 3), # motorbike
    (15, 0), # person
    (16, 2), # pottedplant
    (17, 2), # sheep
    (18, 1), # sofa
    (19, 2), # train
    (20, 2)  # tvmonitor
]

orig_files = os.listdir(root)
cls_files = [osp.join(root, f) for f in orig_files if 'class' in f]
data_files = [osp.join(root, f) for f in orig_files if 'data' in f]
cls_files.sort()
data_files.sort()

# import pdb; pdb.set_trace()
for data_file, cls_file in zip(data_files, cls_files):
    with open(data_file, 'r') as f:
        data_list = f.read().split('\n')[:-1]
    with open(cls_file, 'r') as f:
        fold_n_subclsdata = f.read()
    sub_class_file_list = eval(fold_n_subclsdata)
    img_metadata_classwise = {}
    for sub_cls in sub_class_file_list.keys():
        img_metadata_classwise[sub_cls-1] = [data[0].split('/')[-1].split('.')[0] for data in sub_class_file_list[sub_cls]]
    
    for k, v in img_metadata_classwise.items():
        new_id = categories[k][1]
        sub_class_lists[new_id][categories[k][0]] = sub_class_file_list[k+1]
        for data in data_list:
            name = data.split(' ')[0].split('/')[-1].split('.')[0]
            if name in v:
                data_lists[new_id].append(data)
                
        

for i in range(4):
    with open(osp.join(save_root, 'data_list_%d.txt' % i), 'w') as f:
        f.write('\n'.join(data_lists[i]))
    with open(osp.join(save_root, 'sub_class_file_list_%d.txt' % i), 'w') as f:
        f.write(str(sub_class_lists[i]))

