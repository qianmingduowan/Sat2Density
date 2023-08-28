import torch,os
from torch.utils.data.dataset import Dataset
from PIL import Image
import torchvision.transforms as transforms
import re
from easydict import EasyDict as edict

def data_list(img_root,mode):
    data_list=[]
    if mode=='train':
        split_file=os.path.join(img_root, 'splits/train-19zl.csv')
        with open(split_file) as f:
            list = f.readlines()
            for i in list:
                aerial_name=re.split(r',', re.split('\n', i)[0])[0]
                panorama_name = re.split(r',', re.split('\n', i)[0])[1]
                data_list.append([aerial_name, panorama_name])
    else:
        split_file=os.path.join(img_root+'splits/val-19zl.csv')
        with open(split_file) as f:
            list = f.readlines()
            for i in list:
                aerial_name=re.split(r',', re.split('\n', i)[0])[0]
                panorama_name = re.split(r',', re.split('\n', i)[0])[1]
                data_list.append([aerial_name, panorama_name])
    print('length of dataset is: ', len(data_list))
    return [os.path.join(img_root, i[1]) for i in data_list]
    
def img_read(img,size=None,datatype='RGB'):
    img = Image.open(img).convert('RGB' if datatype=='RGB' else "L")
    if size:
        if type(size) is int:
            size = (size,size)
        img = img.resize(size = size,resample=Image.BICUBIC if datatype=='RGB' else Image.NEAREST)
    img = transforms.ToTensor()(img)
    return img


class Dataset(Dataset):
    def __init__(self, opt,split='train',sub=None,sty_img=None):
        self.pano_list = data_list(img_root=opt.data.root,mode=split)
        if sub:
            self.pano_list = self.pano_list[:sub]
        if opt.task == 'test_vid':
            demo_img_path = os.path.join(opt.data.root, 'streetview/panos', opt.demo_img)
            self.pano_list = [demo_img_path]
        if sty_img:
            assert opt.sty_img.split('.')[-1] == 'jpg'
            demo_img_path = os.path.join(opt.data.root, 'streetview/panos', opt.sty_img)
            self.pano_list = [demo_img_path]

        self.opt = opt

    def __len__(self):
        return len(self.pano_list)

    def __getitem__(self, index):
        pano = self.pano_list[index]
        aer = pano.replace('streetview/panos', 'bingmap/19')
        if self.opt.data.sky_mask:
            sky = pano.replace('streetview/panos','sky_mask').replace('jpg', 'png')
        name = pano
        aer = img_read(aer,  size = self.opt.data.sat_size)
        pano = img_read(pano,size = self.opt.data.pano_size)
        if self.opt.data.sky_mask:
            sky = img_read(sky,size=self.opt.data.pano_size,datatype='L')

        input = {}
        input['sat']=aer
        input['pano']=pano
        input['paths']=name
        if self.opt.data.sky_mask:
            input['sky_mask']=sky
            black_ground = torch.zeros_like(pano)
            if self.opt.data.histo_mode =='grey':
                input['sky_histc'] = (pano*sky+black_ground*(1-sky)).histc()[10:] 
            elif self.opt.data.histo_mode in ['rgb','RGB']:
                input_a  = (pano*sky+black_ground*(1-sky))
                for idx in range(len(input_a)):
                    if idx == 0:
                        sky_histc = input_a[idx].histc()[10:]
                    else:
                        sky_histc = torch.cat([input_a[idx].histc()[10:],sky_histc],dim=0)
                input['sky_histc'] = sky_histc
        return input

