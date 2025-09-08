import torch,os
from torch.utils.data.dataset import Dataset
from PIL import Image
import scipy.io as sio
import torchvision.transforms as transforms
from easydict import EasyDict as edict


def data_list(img_root,mode):
    if mode.lower() == 'train':
        data_list = os.path.join(img_root, 'train.txt')
    elif mode.lower() in ['val']:
        data_list = os.path.join(img_root, 'val.txt')
    elif mode.lower() == 'test':
        data_list = os.path.join(img_root, 'test.txt')
    print('Loading data list from %s' % data_list)
    # load text
    with open(data_list, 'r') as file:
        lines = file.readlines()
    # split ' ' and remove '\n'
    for i in range(len(lines)):
        lines[i] = lines[i].strip('\n')
        lines[i] = lines[i].split(' ')
    return lines


    
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
        # if sty_img:
        #     assert sty_img.endswith('grdView.png')
        #     demo_img_path = os.path.join(opt.data.root,'streetview',sty_img)
        #     self.pano_list = [demo_img_path]

        # elif opt.task in  ['test_vid','test_interpolation'] :
        #     demo_img_path = os.path.join(opt.data.root,'streetview',opt.demo_img.replace('satView_polish.png','grdView.png'))
        #     self.pano_list = [demo_img_path]

        # else:
        self.data_list = data_list(img_root=opt.data.root,mode=split)
        
        self.opt = opt
        self.city_resolution = edict()
        self.city_resolution.Chicago      = 0.111
        self.city_resolution.NewYork      = 0.113
        self.city_resolution.SanFrancisco = 0.118
        self.city_resolution.Seattle      = 0.101

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        aer_path, pano_name, sky, a,b  = data
        aer = img_read(os.path.join(self.opt.data.root,aer_path),  size = self.opt.data.sat_size)
        pano = img_read(os.path.join(self.opt.data.root,pano_name),size = self.opt.data.pano_size)
        name = pano_name
        if self.opt.data.sky_mask:
            sky = img_read(os.path.join(self.opt.data.root,sky),size=self.opt.data.pano_size,datatype='L')

        input = edict()
        input['sat']=aer
        input['pano']=pano
        input['paths']=name
        if self.opt.data.sky_mask:
            input['sky_mask']=sky
            if self.opt.data.histo_mode in ['rgb','RGB']:
                input_a  = (pano*sky)
                sky_histc = []
                for idx in range(len(input_a)):
                    histo = input_a[idx].histc(min=0,max=1)[10:]
                    if histo.sum() != 0:
                        histo = histo/sum(histo)
                    sky_histc.append(histo)
                # from shape [90] to [N* 90]
                sky_histc = torch.concat(sky_histc)
                input['sky_histc'] = sky_histc
            else:
                raise NotImplementedError
        # w and h of the camera position
        input.position = torch.tensor([float(b)/ 320.,float(a)/ 320.])
        return input

if __name__ == '__main__':
    data_list('dataset/vigor','train')