import os.path as osp
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

THIS_PATH = osp.dirname(__file__)
ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '..', '..'))
#IMAGE_PATH = osp.join(ROOT_PATH2, 'data/cub/images')
IMAGE_PATH = osp.join(ROOT_PATH2, '/home/aaa/zll/DFRFS-main-yuanma/DFRFS-main/data/cub/images')
IMAGE_RAW_PATH = osp.join(ROOT_PATH2, 'data/cubraw/images')
#SPLIT_PATH = osp.join(ROOT_PATH2, 'data/cub/split')
SPLIT_PATH = osp.join(ROOT_PATH2, '/home/aaa/zll/DFRFS-main-yuanma/DFRFS-main/data/cub/split')
SPLIT_RAW_PATH = osp.join(ROOT_PATH2, 'data/cub_raw/split')
#############################
BBOX_PATH = '/home/aaa/zll/DFRFS-main-yuanma/DFRFS-main/data/cub/bounding_boxes.txt'
IMAGID_PATH = '/home/aaa/zll/DFRFS-main-yuanma/DFRFS-main/data/cub/images.txt'

class CUB(Dataset):

    def __init__(self, setname, args, augment=False):
        im_size = args.orig_imsize
        txt_path = osp.join(SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(txt_path, 'r').readlines()][1:]
        #####################
        self.image_id_map = self.load_image_id_map(IMAGID_PATH)
        self.bbox_dict = self.load_bounding_boxes(BBOX_PATH)


        self.data, self.label = self.parse_csv(txt_path)
        self.num_class = np.unique(np.array(self.label)).shape[0]
        image_size = 84 #84
        print(f"成功加载 {len(self.image_id_map)} 个图像ID映射。")
        print(f"成功解析 {len(self.data)} 个样本。")

        if augment and setname == 'train':
            transforms_list = [
                  transforms.RandomResizedCrop(image_size),
                  transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                ]
        else:
            transforms_list = [
                  transforms.Resize(92), #92
                  transforms.CenterCrop(image_size),
                  transforms.ToTensor(),
                ]

        # Transformation
        if args.backbone_class == 'ConvNet':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])
        elif args.backbone_class == 'Res12':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
            ])
        elif args.backbone_class == 'Res18':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])       
        elif args.backbone_class == 'Res18_bdc':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])       
        elif args.backbone_class == 'WRN':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])         
        else:
            raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')

    def load_image_id_map(self, image_id_path):
        """加载 image_id 映射，适配没有子目录的情况"""
        image_id_map = {}
        with open(image_id_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split(' ')
                if len(parts) != 2:
                    continue
                image_id, image_path = parts

                # 只保留文件名，去除子目录
                image_name = osp.basename(image_path)

                # 映射 image_name 到 image_id
                image_id_map[image_name] = int(image_id)
        print(f"成功加载 {len(image_id_map)} 个 image_id 映射。")
        return image_id_map

    def load_bounding_boxes(self, bbox_path):
        bbox_dict = {}
        with open(bbox_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                image_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])
                bbox_dict[image_id] = (x, y, w, h)
        return bbox_dict

    def parse_csv(self, txt_path):
        data = []
        label = []
        lb = -1
        self.wnids = []
        lines = [x.strip() for x in open(txt_path, 'r').readlines()][1:]

        for l in lines:
            context = l.split(',')
            name = context[0] 
            wnid = context[1]
            path = osp.join(IMAGE_PATH, name)
            ################################
            # 构造图像路径（无子目录）
            image_path = osp.join(IMAGE_PATH, name)

            # 查找 image_id
            image_id = self.image_id_map.get(name, None)
            if image_id is None:
                print(f"未找到 {name} 对应的 image_id，跳过。")
                continue

            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            ######################################
            # 如果边界框存在该 image_id，则加入数据集
            if image_id in self.bbox_dict:
                data.append((image_path, self.bbox_dict[image_id]))
                label.append(lb)
                
            #data.append(path)
            #label.append(lb)

        return data, label


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        #data, label = self.data[i], self.label[i]
        #image = self.transform(Image.open(data).convert('RGB'))
        ############################################################
        (image_path, bbox), label = self.data[i], self.label[i]
        image = Image.open(image_path).convert('RGB')
        x, y, w, h = bbox
        image = image.crop((x, y, x + w, y + h))
        image = self.transform(image)
        return image, label            


class CUB_RAW(Dataset):
    
    def __init__(self, setname, args, augment=False):
        txt_path = osp.join(SPLIT_RAW_PATH, setname + '.csv')
        
        self.data, self.label = self.parse_csv(txt_path, setname)
        
        self.num_class = np.unique(np.array(self.label)).shape[0]
        image_size = 84
        
        if augment and setname == 'train':
            transforms_list = [
                  transforms.RandomResizedCrop(image_size),
                  transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                ]
        else:
            transforms_list = [
                  transforms.Resize(92),
                  transforms.CenterCrop(image_size),
                  transforms.ToTensor(),
                ]

        # Transformation
        if args.backbone_class == 'ConvNet':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])
        elif args.backbone_class == 'Res12':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
            ])
        elif args.backbone_class == 'Res18':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])                  
        else:
            raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')

    def parse_csv(self, txt_path, setname):
        data = []
        label = []
        lb = -1
        self.wnids = []
        lines = [x.strip() for x in open(txt_path, 'r').readlines()][0:]

        for l in lines:
            context = l.split(',')
            name = context[0] 
            wnid = context[1]
            path = osp.join(IMAGE_RAW_PATH, setname, wnid, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
                
            data.append(path)
            label.append(lb)

        return data, label


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label = self.data[i], self.label[i]
        image = self.transform(Image.open(data).convert('RGB'))
        return image, label  