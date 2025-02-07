import os
import os.path as osp
import ujson

root = '/root/sawsam/data/splits/lists/coco/fss_list/train'
save_root = '/root/sawsam/data/splits/lists/coco_all/fss_list/train'
data_lists = []
sub_class_lists = {}

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
    sub_class_lists.update(sub_class_file_list)
    data_lists.extend(data_list)
                
        

with open(osp.join(save_root, 'data_list_all.txt'), 'w') as f:
    f.write('\n'.join(data_lists))
with open(osp.join(save_root, 'sub_class_file_list_all.txt'), 'w') as f:
    f.write(str(sub_class_lists))

