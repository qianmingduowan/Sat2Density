# from this import d
import torch
from .base_model import BaseModel
import importlib
from  torch.utils.data import DataLoader
from easydict import EasyDict as edict

class Model(BaseModel):
    def __init__(self, opt, wandb=None):

        """Initialize the Generator.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt,wandb)
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
            self.real_A: aerial images
            self.real_B: ground images
            self.image_paths: images paths of ground images
            self.sky_mask: the sky mask of ground images
            self.sky_histc: the histogram of selected sky
        """     
        self.real_A = input['sat' ].to(self.device)
        self.real_B = input['pano'].to(self.device) if 'pano' in input else None # for testing
        self.image_paths = input['paths']
        if self.opt.data.sky_mask:
            self.sky_mask = input['sky_mask'].to(self.device) if 'sky_mask' in input else None # for testing
        if self.opt.data.histo_mode and self.opt.data.sky_mask:
            self.sky_histc = input['sky_histc'].to(self.device) if 'sky_histc' in input else None # for testing
        else: self.sky_histc = None

    def forward(self,opt):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # origin_H_W is the inital localization of camera
        if opt.task != 'test_vid':
            opt.origin_H_W=None
        if hasattr(opt.arch.gen,'style_inject'):
            # replace the predicted sky with selected sky histogram
            if opt.arch.gen.style_inject == 'histo':
                self.out_put =  self.netG(self.real_A,self.sky_histc.detach(),opt) 
            else:
                raise Exception('Unknown style inject mode')
        else:
            self.out_put =  self.netG(self.real_A,None,opt) 
        self.out_put = edict(self.out_put)
        self.fake_B = self.out_put.pred
        # perceptive image

    def backward_D(self,opt):
        """Calculate GAN loss for the discriminator"""
        self.optimizer_D.zero_grad()
        self.netG.eval()
        with torch.no_grad():
            self.forward(opt)                   
            self.out_put.pred = self.out_put.pred.detach()
        net_D_output = self.netD(self.real_B, self.out_put)

        output_fake = self._get_outputs(net_D_output, real=False)
        output_real = self._get_outputs(net_D_output, real=True)
        fake_loss = self.criteria['GAN'](output_fake, False, dis_update=True)
        true_loss = self.criteria['GAN'](output_real, True, dis_update=True)
        self.dis_losses = dict()
        self.dis_losses['GAN/fake'] = fake_loss
        self.dis_losses['GAN/true'] = true_loss
        self.dis_losses['DIS'] = fake_loss + true_loss
        self.dis_losses['DIS'].backward()
        self.optimizer_D.step()          


    def backward_G(self,opt):
        self.optimizer_G.zero_grad()       
        self.loss = {}
        self.netG.train()
        self.forward(opt) 
        net_D_output = self.netD(self.real_B, self.out_put) 
        pred_fake = self._get_outputs(net_D_output, real=False)
        self.loss['GAN'] = self.criteria['GAN'](pred_fake, True, dis_update=False)
        if 'GaussianKL' in self.criteria:
            self.loss['GaussianKL'] = self.criteria['GaussianKL'](self.out_put['mu'], self.out_put['logvar'])
        if 'L1' in self.criteria:
            self.loss['L1'] = self.criteria['L1'](self.real_B,self.fake_B)
        if 'L2' in self.criteria:
            self.loss['L2'] = self.criteria['L2'](self.real_B,self.fake_B)
        if 'SSIM' in self.criteria:
            self.loss['SSIM'] = 1-self.criteria['SSIM'](self.real_B, self.fake_B)
        if 'GaussianKL' in self.criteria:
            self.loss['GaussianKL'] = self.criteria['GaussianKL'](self.out_put['mu'], self.out_put['logvar'])
        if 'sky_inner' in self.criteria:
            self.loss['sky_inner'] = self.criteria['sky_inner'](self.out_put.opacity, 1-self.sky_mask)
        if 'Perceptual' in self.criteria:
            self.loss['Perceptual'] = self.criteria['Perceptual'](self.fake_B,self.real_B)
        if 'feature_matching' in self.criteria:
            self.loss['feature_matching']  = self.criteria['feature_matching'](net_D_output['fake_features'], net_D_output['real_features'])
        self.loss_G = 0
        for key in self.loss:
            self.loss_G += self.loss[key] * self.weights[key]
        self.loss['total'] = self.loss_G 
        self.loss_G.backward()
        self.optimizer_G.step()             # udpate G's weights


    def load_dataset(self,opt):
        data = importlib.import_module("data.{}".format(opt.data.dataset))
        if opt.task in ["train", "Train"]:
            train_data = data.Dataset(opt,"train",opt.data.train_sub)
            
            self.train_loader = DataLoader(train_data,batch_size=opt.batch_size,shuffle=True,num_workers=opt.data.num_workers,drop_last=True)
            self.len_train_loader = len(self.train_loader)

        val_data   = data.Dataset(opt,"val")
        opt.batch_size = 1 if opt.task in ["test" , "val","vis_test",'test_vid','test_sty'] else opt.batch_size
        opt.batch_size = 1 if opt.task=='test_speed' else opt.batch_size
        self.val_loader = DataLoader(val_data,batch_size=opt.batch_size,shuffle=False,num_workers=opt.data.num_workers)
        self.len_val_loader   = len(self.val_loader)
        # you can select one random image as a style of all predicted skys
        # if None, we use the corresponding style of GT 
        if opt.sty_img:
            sty_data = data.Dataset(opt,sty_img = opt.sty_img)
            self.sty_loader = DataLoader(sty_data,batch_size=1,num_workers=1,shuffle=False)
        # The followings are only used for test the illumination interpolation.
        if opt.sty_img1:
            sty1_data = data.Dataset(opt,sty_img = opt.sty_img1)
            self.sty_loader1 = DataLoader(sty1_data,batch_size=1,num_workers=1,shuffle=False)
        if opt.sty_img2:
            sty2_data = data.Dataset(opt,sty_img = opt.sty_img2)
            self.sty_loader2 = DataLoader(sty2_data,batch_size=1,num_workers=1,shuffle=False)

    def build_networks(self, opt):
        if 'imaginaire' in opt.arch.gen.netG:
            lib_G = importlib.import_module(opt.arch.gen.netG)
            self.netG = lib_G.Generator(opt).to(self.device)
        else:
            raise Exception('Unknown discriminator function')

        if opt.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            if opt.arch.dis.netD == 'imaginaire.discriminators.multires_patch_pano':
                lib_D = importlib.import_module(opt.arch.dis.netD)
                self.netD = lib_D.Discriminator(opt.arch.dis).to(self.device)
            else:
                raise Exception('Unknown discriminator function')
