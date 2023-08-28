from unittest import result
from matplotlib.pyplot import hist
from torch.utils import data
from torch.utils.data.dataset import Dataset

import os,torch
from PIL import Image
import torchvision.transforms as T
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import torch.nn.functional as F
from imaginaire.evaluation.segmentation import get_segmentation_hist_model,get_miou,compute_hist
import lpips
from easydict import EasyDict as edict
from tqdm import tqdm
import piq
from  torch.utils.data import DataLoader
from piq import FID,KID
import numpy as np

result_path = 'result/Ours-pers-sin-sty'
gt_path = 'dataset/CVACT/streetview_test'


class Dataset_img(Dataset):
    def __init__(self, dir):
        self.dir = dir
        self.datalist = sorted(os.listdir(dir))
    
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        img = os.path.join(self.dir,self.datalist[index])
        img = Image.open(img).convert('RGB')
        img = T.ToTensor()(img)
        return {'images':img}



data_gt = Dataset_img(gt_path)
data_pred = Dataset_img(result_path)


loss_fn_alex = lpips.LPIPS(net='alex',eval_mode=True).cuda()
loss_fn_squeeze = lpips.LPIPS(net='squeeze',eval_mode=True).cuda()


data_list = os.listdir(result_path)
results = edict()
results.psnr = []
results.ssim = []
results.alex = []
results.squeeze = []
results.RMSE  = []

dataloader_pred = DataLoader(data_pred,batch_size=1,shuffle=False,num_workers=10)
dataloader_gt   = DataLoader(data_gt,batch_size=1,shuffle=False,num_workers=10)
for i in tqdm(zip(dataloader_pred,dataloader_gt),ncols=100):
    pred = i[0]['images'].cuda()
    gt   = i[1]['images'].cuda()
    results.psnr.append(-10*F.mse_loss(pred,gt).log10().item())
    results.ssim.append(ssim(pred, gt,data_range=1.).item())
    results.alex.append(torch.mean(loss_fn_alex((pred*2.)-1, (2.*gt)-1)).cpu().item())
    results.squeeze.append(torch.mean(loss_fn_squeeze((pred*2.)-1, (2.*gt)-1)).cpu().item())
    results.RMSE.append(torch.sqrt(F.mse_loss(pred,gt)).item()*255)

for i in results:
    print("%-10s"%i, ':',np.mean(results[i]))
