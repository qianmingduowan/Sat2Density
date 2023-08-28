import torch,os
from torch.utils.data.dataset import Dataset
from PIL import Image
import scipy.io as sio
import torchvision.transforms as transforms

def data_list(img_root,mode):
    exist_aer_list = os.listdir(os.path.join(img_root , 'satview_correct'))
    exist_grd_list = os.listdir(os.path.join(img_root , 'streetview'))
    allDataList = os.path.join(img_root, 'ACT_data.mat')
    anuData = sio.loadmat(allDataList)

    all_data_list = []
    for i in range(0, len(anuData['panoIds'])):
        grd_id_align = anuData['panoIds'][i] + '_grdView.png'
        sat_id_ori = anuData['panoIds'][i] + '_satView_polish.png'
        all_data_list.append([grd_id_align, sat_id_ori])

    data_list = []
    
    if mode=='train':
        training_inds = anuData['trainSet']['trainInd'][0][0] - 1
        trainNum = len(training_inds)
        for k in range(trainNum):
            data_list.append(all_data_list[training_inds[k][0]])
    else:
        val_inds = anuData['valSet']['valInd'][0][0] - 1
        valNum = len(val_inds)
        for k in range(valNum):
            data_list.append(all_data_list[val_inds[k][0]])


    pano_list = [img_root + 'streetview/' + item[0] for item in data_list if item[0] in exist_grd_list and item[1] in exist_aer_list]

    return pano_list
    
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
        if sty_img:
            assert sty_img.endswith('grdView.png')
            demo_img_path = os.path.join(opt.data.root,'streetview',sty_img)
            self.pano_list = [demo_img_path]

        elif opt.task in  ['test_vid','test_interpolation'] :
            demo_img_path = os.path.join(opt.data.root,'streetview',opt.demo_img.replace('satView_polish.png','grdView.png'))
            self.pano_list = [demo_img_path]

        else:
            self.pano_list = data_list(img_root=opt.data.root,mode=split)
            if sub:
                self.pano_list = self.pano_list[:sub]
        
        # select some ground images to test the influence of different skys.
        # different skys guide different illumination intensity, colors, and etc.
        if opt.task == 'test_sty':
            demo_name = [
                'dataset/CVACT/streetview/pPfo7qQ1fP_24rXrJ2Uxog_grdView.png',
                'dataset/CVACT/streetview/YL81FiK9PucIvAkr1FHkpA_grdView.png',
                'dataset/CVACT/streetview/Tzis1jBKHjbXiVB2oRYwAQ_grdView.png',
                'dataset/CVACT/streetview/eqGgeBLGXRhSj6c-0h0KoQ_grdView.png',
                'dataset/CVACT/streetview/pdZmLHYEhe2PHj_8-WHMhw_grdView.png',
                'dataset/CVACT/streetview/ehsu9Q3iTin5t52DM-MwyQ_grdView.png',
                'dataset/CVACT/streetview/agLEcuq3_-qFj7wwGbktVg_grdView.png',
                'dataset/CVACT/streetview/HwQIDdMI3GfHyPGtCSo6aA_grdView.png',
                'dataset/CVACT/streetview/hV8svb3ZVXcQ0AtTRFE1dQ_grdView.png',
                'dataset/CVACT/streetview/fzq2mBfKP3UIczAd9KpMMg_grdView.png',
                'dataset/CVACT/streetview/acRP98sACUIlwl2ZIsEyiQ_grdView.png',
                'dataset/CVACT/streetview/WSh9tNVryLdupUlU0ri2tQ_grdView.png',
                'dataset/CVACT/streetview/FhEuB9NA5o08VJ_TBCbHjw_grdView.png',
                'dataset/CVACT/streetview/YHfpn2Mgu1lqgT2OUeBpOg_grdView.png',
                'dataset/CVACT/streetview/vNhv7ZP1dUkJ93UwFXagJw_grdView.png',
            ]
            self.pano_list = demo_name

        self.opt = opt

    def __len__(self):
        return len(self.pano_list)

    def __getitem__(self, index):
        pano = self.pano_list[index]
        aer = pano.replace('streetview','satview_correct').replace('_grdView','_satView_polish')
        if self.opt.data.sky_mask:
            sky = pano.replace('streetview','pano_sky_mask')
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

