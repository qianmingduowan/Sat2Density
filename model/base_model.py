import os
import torch
from abc import ABC, abstractmethod
import wandb
import options
import utils
from pytorch_msssim import ssim, SSIM
import numpy as np
import torchvision
from tqdm import tqdm
import lpips
from imaginaire.losses import FeatureMatchingLoss, GaussianKLLoss, PerceptualLoss,GANLoss
import cv2
from imaginaire.utils.trainer import get_scheduler
from .geometry_transform import render_sat
from model import geometry_transform
import csv



class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt,wandb=None):
        """Initialize the BaseModel class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
        """
        self.wandb = wandb
        if opt.isTrain:
            opt.save_dir =wandb.dir
            options.save_options_file(opt,opt.save_dir)
        self.opt = opt
        self.device = "cpu" if opt.cpu or not torch.cuda.is_available() else "cuda:{}".format(opt.gpu)
        # torch.backends.cudnn.benchmark = True
        self.model_names = []
        self.train_loader = None
        self.val_loader = None 
        self.sty_loader = None
        self.loss_fn_alex = lpips.LPIPS(net='alex',eval_mode=True).cuda()
        if opt.task=='test':
            self.loss_fn_sque = lpips.LPIPS(net='squeeze',eval_mode=True).cuda()
        self.mseloss = torch.nn.MSELoss(True,True)
        self.criteria = {}
        self.weights = {}
        if hasattr(opt.optim.loss_weight, 'GaussianKL'):
            if opt.optim.loss_weight.GaussianKL:
                self.criteria['GaussianKL'] = GaussianKLLoss()
                self.weights['GaussianKL'] = opt.optim.loss_weight.GaussianKL
        if hasattr(opt.optim.loss_weight, 'L1'):
            if opt.optim.loss_weight.L1:
                self.criteria['L1']  = torch.nn.L1Loss(True,True)
                self.weights['L1'] = opt.optim.loss_weight.L1
        if hasattr(opt.optim.loss_weight, 'L2'):
            if opt.optim.loss_weight.L2: 
                self.criteria['L2'] = torch.nn.MSELoss(True,True)
                self.weights['L2'] = opt.optim.loss_weight.L2
        if hasattr(opt.optim.loss_weight, 'SSIM'):
            if opt.optim.loss_weight.SSIM: 
                self.criteria['SSIM'] = SSIM(data_range =1., size_average=True, channel=3)
                self.weights['SSIM']  = opt.optim.loss_weight.SSIM
        if hasattr(opt.optim.loss_weight, 'Perceptual'):
            if opt.optim.loss_weight.Perceptual: 
                self.criteria['Perceptual'] = \
                    PerceptualLoss(
                        network=opt.optim.perceptual_loss.mode,
                        layers=opt.optim.perceptual_loss.layers,
                        weights=opt.optim.perceptual_loss.weights).to(self.device)
                self.weights['Perceptual'] = opt.optim.loss_weight.Perceptual
        if hasattr(opt.optim.loss_weight, 'sky_inner'):
            if opt.optim.loss_weight.sky_inner:
                self.criteria['sky_inner'] = torch.nn.L1Loss(True,True)
                self.weights['sky_inner'] = opt.optim.loss_weight.sky_inner
        if hasattr(opt.optim.loss_weight, 'feature_matching'):
            if opt.optim.loss_weight.feature_matching:
                self.criteria['feature_matching'] = FeatureMatchingLoss()
                self.weights['feature_matching'] = opt.optim.loss_weight.feature_matching
        self.weights['GAN'] = opt.optim.loss_weight.GAN
        self.criteria['GAN'] = GANLoss(gan_mode=opt.optim.gan_mode)


    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    def save_checkpoint(self,ep=0,latest=False):
        """
        save trained models.
        Args:
            ep (int, optional): model epochs. Defaults to 0.
            latest (bool, optional): qhether it is the latest model. Defaults to False.
        """        
        ckpt_save_path = os.path.join(self.wandb.dir,'checkpoint')
        if not os.path.exists(ckpt_save_path):
            os.mkdir(ckpt_save_path)
        utils.save_checkpoint(self,ep=ep,latest=latest,output_path=ckpt_save_path)
        if not latest:
            print("checkpoint saved: {0}, epoch {1} ".format(self.opt.name,ep))



    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)


    def setup_optimizer(self,opt):
        # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.optim.lr_gen, betas=(opt.optim.beta1, 0.999),eps=1.e-7)
        if opt.isTrain:
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.optim.lr_dis, betas=(opt.optim.beta1, 0.999))
        if opt.optim.lr_policy:
            self.sch_G = get_scheduler(opt.optim, self.optimizer_G)
            self.sch_D = get_scheduler(opt.optim, self.optimizer_D)

    def optimize_parameters(self,opt):
        self.netG.train()
        # update Discriminators
        self.backward_D(opt)                # calculate gradients for D

        # update Generator
        self.backward_G(opt)                   # calculate graidents for G

        psnr1 = -10*self.mseloss(self.fake_B.detach(),self.real_B.detach()).log10().item()
        ssim_ = ssim(self.real_B.detach().float(), self.fake_B.detach().float(),data_range=1.)

        out_dict = {
                        "train_ssim": ssim_,
                        "train_psnr1": psnr1,
                    }
        # adjust learning rates according to schedule 
        if opt.optim.lr_policy:
            out_dict["lr_D"]=self.sch_D.get_lr()[0]
            out_dict["lr_G"]=self.sch_G.get_lr()[0]
        out_dict.update(self.loss)
        out_dict.update(self.dis_losses)
        self.wandb.log(out_dict)

    def validation(self,opt):
        """Used for validation and test in Center Ground-View Synthesis setting

        Args:
            opt (_type_): option dict
        """        
        print(10*"*","validate",10*"*")
        self.netG.eval()
        # six image reconstruction metrics
        psnr_val = []
        ssim_val = []
        lpips_ale_val = []
        lpips_squ_val = []
        rmse_val = []
        sd_val = []
        with torch.no_grad():
            # set the sky of all images with predefined sky histogram.
            if opt.sty_img:
                for _,data in enumerate(self.sty_loader):
                    self.set_input(data)
                    self.style_temp=self.sky_histc
                    break

            for _,data in enumerate(tqdm(self.val_loader,ncols=100)):
                self.set_input(data)
                # if true: use the sky of predefined image
                # if false: use the sky of corresponding GT
                if opt.sty_img:
                    self.sky_histc = self.style_temp
                
                self.forward(opt)
                rmse = torch.sqrt(self.mseloss(self.fake_B*255.,self.real_B*255.)).item()
                sd = sd_func(self.real_B,self.fake_B)
                rmse_val.append(rmse)
                sd_val.append(sd)

                psnr1 = -10*self.mseloss(self.fake_B,self.real_B).log10().item()
                ssim_ = ssim(self.real_B, self.fake_B,data_range=1.).item()
                lpips_ale = torch.mean(self.loss_fn_alex((self.real_B*2.)-1, (2.*self.fake_B)-1)).cpu()
                if opt.task=='test':
                    lpips_sque = torch.mean(self.loss_fn_sque((self.real_B*2.)-1, (2.*self.fake_B)-1)).cpu()
                    lpips_squ_val.append(lpips_sque)
                psnr_val.append(psnr1)
                ssim_val.append(ssim_)
                lpips_ale_val.append(lpips_ale)
                    
                if opt.task in ['vis_test']:
                    if not os.path.exists(opt.vis_dir):
                        os.mkdir(opt.vis_dir)

                    sat_opacity,sat_depth = render_sat(opt,self.out_put['voxel'])

                    self.out_put['depth'] = (self.out_put['depth']/self.out_put['depth'].max())*255.
                    sat_depth = (sat_depth/sat_depth.max())*255.
                    for i in range(len(self.fake_B)):
                        depth_save  = cv2.applyColorMap(self.out_put['depth'][i].squeeze().cpu().numpy().astype(np.uint8), cv2.COLORMAP_TURBO)
                        depth_sat_save = cv2.applyColorMap(sat_depth[i].squeeze().cpu().numpy().astype(np.uint8), cv2.COLORMAP_TURBO)
                        # cat generated ground images, GT ground images, predicted ground depth
                        torchvision.utils.save_image([self.fake_B[i].cpu(),self.real_B[i].cpu(),torch.flip(torch.from_numpy(depth_save).permute(2,0,1)/255.,[0])],os.path.join(opt.vis_dir,os.path.basename(self.image_paths[i])))
                        # cat GT satellite images, predicted satellite depth
                        torchvision.utils.save_image( [self.real_A[i].cpu() ,torch.flip(torch.from_numpy(depth_sat_save).permute(2,0,1)/255.,[0])],os.path.join(opt.vis_dir,os.path.basename(self.image_paths[i]).rsplit('.', 1)[0]+'_sat.jpg'))
                        # ground opacity
                        torchvision.utils.save_image([self.out_put['opacity'][i]] ,os.path.join(opt.vis_dir,os.path.basename(self.image_paths[i]).rsplit('.', 1)[0]+'_sat.jpg'))
        psnr_avg = np.average(psnr_val)
        ssim_avg = np.average(ssim_val)

        lpips_ale_avg = np.average(lpips_ale_val)
        if 'test' in opt.task:
            lpips_squ_avg = np.average(lpips_squ_val)

        rmse_avg = np.average(rmse_val)
        sd_avg = np.average(sd_val)
        if opt.task in ["train" , "Train"]:
            out_dict =   {
                            'val_psnr': psnr_avg,
                            'val_ssim': ssim_avg,
                            'val_lpips_ale':lpips_ale_avg,
                            'val_rmse':rmse_avg,
                            'val_sd':sd_avg
                            }  
            if opt.task=='test':
                out_dict['val_lpips_squ'] =  lpips_squ_avg          
            self.wandb.log(out_dict,commit=False)
        else:
            print(
                {
                'val_rmse':rmse_avg,
                'val_ssim': ssim_avg,
                'val_psnr': psnr_avg,
                'val_sd':sd_avg,
                'val_lpips_ale':lpips_ale_avg,
                'val_lpips_squ':lpips_squ_avg,
                }
                )
            with open('test_output.csv', mode='a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([rmse_avg, ssim_avg, psnr_avg, sd_avg, lpips_ale_avg, lpips_squ_avg])
                
    def test_vid(self,opt):
        """Used for synthesis ground video

        Args:
            opt (_type_): option dict
        """        
        ckpt_list = os.listdir('wandb/')
        for i in ckpt_list:
            if opt.test_ckpt_path in i:
                ckpt_path = i
        
        ckpt = torch.load(os.path.join('wandb/',ckpt_path,'files/checkpoint/model.pth'))['netG']
        print('load success!')
        self.netG.load_state_dict(ckpt,strict=True)
        self.netG.eval()
        print(10*"*","test_video",10*"*")


        pixels = []
        if os.path.exists('vis_video/pixels.csv'):

            with open('vis_video/pixels.csv', 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    x = float(row['x']) #x is 
                    y = float(row['y'])
                    pixels.append((x, y))
        else:
            print('only render center point without vis_video/pixels.csv')
            pixels = [(128,128)]

        if opt.sty_img:
            # inference with illumination from other images
            for idx,data in enumerate(self.sty_loader):
                self.set_input(data)
                self.style_temp=self.sky_histc
                break
        with torch.no_grad():
            for idx,data in enumerate(self.val_loader):
                self.set_input(data)
                if opt.sty_img:
                    self.sky_histc = self.style_temp
                for i,(x,y) in enumerate(pixels):
                    opt.origin_H_W = [(y-128)/128 , (x-128)/128]
                    print(opt.origin_H_W)
                    self.forward(opt)



                    if not os.path.exists('vis_video'):
                        os.mkdir('vis_video')

                    # save voxel to visalize & satellite depth, works well on cvact
                    if i==0:
                        # pre-process for better visualize
                        volume_data = self.out_put.voxel.squeeze().cpu().numpy().transpose((1,2,0))
                        volume_data = np.clip(volume_data, None, 10)

                        import pyvista as pv

                        grid = pv.UniformGrid()
                        grid.dimensions = volume_data.shape
                        grid.spacing = (1, 1, 1)
                        grid.origin = (0, 0, 0)
                        grid.point_data['values'] = volume_data.flatten(order='F')
                        grid.save(os.path.join('vis_video',"volume_data.vtk") ) # vtk file could be visualized by ParaView app

                        sat_opacity,sat_depth = render_sat(opt,self.out_put['voxel'])
                        sat_depth = (2 - sat_depth)/(opt.data.max_height/15)*255.
                        depth_sat_save = cv2.applyColorMap(sat_depth[0].squeeze().cpu().numpy().astype(np.uint8), cv2.COLORMAP_TURBO)
                        torchvision.utils.save_image(torch.flip(torch.from_numpy(depth_sat_save).permute(2,0,1)/255.,[0]) ,os.path.join('vis_video',os.path.basename(self.image_paths[0])).replace('.png','_satdepth.png'))
                        torchvision.utils.save_image( [self.real_A[0].cpu() ]                      ,os.path.join('vis_video',os.path.basename(self.image_paths[0]).replace('.png','_sat.png')))
                        torchvision.utils.save_image( [self.real_B[0].cpu() ]                      ,os.path.join('vis_video',os.path.basename(self.image_paths[0]).replace('.png','_pano.png')))
                        
                    self.out_put['depth'] = (self.out_put['depth']/self.out_put['depth'].max())*255.
                    depth_save  = cv2.applyColorMap(self.out_put['depth'][0].squeeze().cpu().numpy().astype(np.uint8), cv2.COLORMAP_TURBO)
                    depth_save = torch.flip(torch.from_numpy(depth_save).permute(2,0,1)/255.,[0])

                    
                    save_img = self.out_put.pred[0].cpu()
                    name = '%05d' % int(i) + ".png"
                    torchvision.utils.save_image(save_img,os.path.join('vis_video',os.path.basename(self.image_paths[0])).replace('.png',name))

                    save_img = depth_save
                    name = '%05d' % int(i) + "_depth.png"
                    torchvision.utils.save_image(save_img,os.path.join('vis_video',os.path.basename(self.image_paths[0])).replace('.png',name))

                    # save_img = self.out_put.generator_inputs[0][:3,:,:]
                    # name = '%05d' % int(i) + "_color_project.png"
                    # torchvision.utils.save_image(save_img,os.path.join('vis_video',os.path.basename(self.image_paths[0])).replace('.png',name))

    def test_interpolation(self,opt):
        """Used for test interpolation

        Args:
            opt (_type_): option dict
        """        
        ckpt_list = os.listdir('wandb/')
        for i in ckpt_list:
            if opt.test_ckpt_path in i:
                ckpt_path = i
        
        ckpt = torch.load(os.path.join('wandb/',ckpt_path,'files/checkpoint/model.pth'))['netG']
        print('load success!')
        self.netG.load_state_dict(ckpt,strict=True)
        self.netG.eval()

        pixels = [(128,128)]
        if opt.sty_img1:
            for idx,data in enumerate(self.sty_loader1):
                self.set_input(data)
                self.style_temp1=self.sky_histc
                break
        if opt.sty_img2:
            for idx,data in enumerate(self.sty_loader2):
                self.set_input(data)
                self.style_temp2=self.sky_histc
                break
        
        with torch.no_grad():
            for idx,data in enumerate(self.val_loader):
                self.set_input(data)
                self.sky_histc1 = self.style_temp1
                self.sky_histc2 = self.style_temp2
                x,y =  pixels[0]
                opt.origin_H_W = [(y-128)/128 , (x-128)/128]
                print(opt.origin_H_W)
                    

                estimated_height = self.netG.depth_model(self.real_A)
                geo_outputs = geometry_transform.render(opt,self.real_A,estimated_height,self.netG.pano_direction,PE=self.netG.PE)
                generator_inputs,opacity,depth = geo_outputs['rgb'],geo_outputs['opacity'],geo_outputs['depth']
                if self.netG.gen_cfg.cat_opa:
                    generator_inputs = torch.cat((generator_inputs,opacity),dim=1)
                if self.netG.gen_cfg.cat_depth:
                    generator_inputs = torch.cat((generator_inputs,depth),dim=1)
                _, _, z1 = self.netG.style_encode(self.sky_histc1)
                _, _, z2 = self.netG.style_encode(self.sky_histc2)
                num_inter = 60
                for i in range(num_inter):
                    z = z1 * (1-i/(num_inter-1)) + z2* (i/(num_inter-1))
                    z = self.netG.style_model(z)
                    output_RGB = self.netG.denoise_model(generator_inputs,z)

                    save_img = output_RGB.cpu()
                    name = 'img{:03d}.png'.format(i)
                    if not os.path.exists('vis_interpolation'):
                        os.mkdir('vis_interpolation')
                    torchvision.utils.save_image(save_img,os.path.join('vis_interpolation',name))



                        
                  

    def test_speed(self,opt):
        self.netG.eval()
        random_input = torch.randn(1, 3, 256, 256).to(opt.device)
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        iterations  = 300

        times = torch.zeros(iterations)
        with torch.no_grad():
            for _ in range(50):
                _ = self.netG(random_input,None,opt)
            for iter in range(iterations):
                starter.record()
                _ = self.netG(random_input,None,opt)
                ender.record()
                torch.cuda.synchronize() 
                curr_time = starter.elapsed_time(ender) # 计算时间
                times[iter] = curr_time
        # print(curr_time)

        mean_time = times.mean().item()
        print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000/mean_time))


    def test_sty(self,opt):
        ckpt_list = os.listdir('wandb/')
        for i in ckpt_list:
            if opt.test_ckpt_path in i:
                ckpt_path = i
        
        ckpt = torch.load(os.path.join('wandb/',ckpt_path,'files/checkpoint/model.pth'))['netG']
        print('load success!')
        self.netG.load_state_dict(ckpt,strict=True)
        self.netG.eval()
        print(10*"*","test_sty",10*"*")
        self.netG.eval()
        self.style_temp_list = []
        with torch.no_grad():
            num_val_loader = len(self.val_loader)
            for i in range(num_val_loader):
                for idx,data in enumerate(tqdm(self.val_loader,ncols=100)):
                    self.set_input(data)
                    
                    if i==0:
                        self.style_temp_list.append(self.sky_histc)
                        name = '%05d' % int(idx)
                        torchvision.utils.save_image( [self.real_A[0].cpu() ]  ,os.path.join(opt.vis_dir,os.path.basename(self.image_paths[0]).replace('.png',name+'_sat.png')))
                    self.sky_histc = self.style_temp_list[i]
                    self.forward(opt)
                    if not os.path.exists(opt.vis_dir):
                        os.mkdir(opt.vis_dir)
                    name = '%05d' % int(idx)+'_'+'%05d' % int(i)
                    name= name+ '.png'
                    torchvision.utils.save_image(self.fake_B[0].cpu(),os.path.join(opt.vis_dir, name))

    def train(self,opt):
        self.validation(opt)
        for current_epoch in range(opt.max_epochs):
            print(10*'-','current epoch is ',current_epoch,10*'-')
            for idx,data in enumerate(tqdm(self.train_loader,ncols=100)):
                self.set_input(data)
                self.optimize_parameters(opt)
                if idx%500==0 :
                    out_ing_dict = {
                                    'train_input': wandb.Image(self.real_A[0].float()),
                                    'train_pred_and_gt': wandb.Image(torch.cat([self.fake_B,self.real_B],2)[0].float()),
                                    }
                    if hasattr(self.out_put, 'inter_RGB'):
                        out_ing_dict["train_inner_pred"] = wandb.Image(self.out_put.inter_RGB[0].float())
                    if opt.arch.gen.transform_mode in ['volum_rendering']:
                        out_ing_dict['train_inner_opacity'] = wandb.Image(self.out_put.opacity[0].float())
                    self.wandb.log(out_ing_dict,commit=False)
                if  opt.optim.lr_policy.iteration_mode:
                    self.sch_G.step()
                    self.sch_D.step()
            if not opt.optim.lr_policy.iteration_mode:
                self.sch_G.step()
                self.sch_D.step()
            self.validation(opt)
            if current_epoch%5==0:
                self.save_checkpoint(ep=current_epoch)
        self.save_checkpoint(ep=current_epoch)

    def test(self,opt):
        ckpt_list = os.listdir('wandb/')
        for i in ckpt_list:
            if '.zip' not in i:
                if opt.test_ckpt_path in i:
                    ckpt_path = i
        
        ckpt = torch.load(os.path.join('wandb/',ckpt_path,'files/checkpoint/model.pth'))['netG']
        print('load success!')
        self.netG.load_state_dict(ckpt,strict=True)
        # print(10*"*","validate",10*"*")
        self.validation(opt)
        print('if --task=vis_test,visible results will be saved,you can add "--vis_dir=xxx" to save in other dictionary',opt.vis_dir)


    def _get_outputs(self, net_D_output, real=True):
        r"""Return output values. Note that when the gan mode is relativistic.
        It will do the difference before returning.

        Args:
           net_D_output (dict):
               real_outputs (tensor): Real output values.
               fake_outputs (tensor): Fake output values.
           real (bool): Return real or fake.
        """

        def _get_difference(a, b):
            r"""Get difference between two lists of tensors or two tensors.

            Args:
                a: list of tensors or tensor
                b: list of tensors or tensor
            """
            out = list()
            for x, y in zip(a, b):
                if isinstance(x, list):
                    res = _get_difference(x, y)
                else:
                    res = x - y
                out.append(res)
            return out

        if real:
            return net_D_output['real_outputs']
        else:
            return net_D_output['fake_outputs']


def sd_func(real, fake):
    '''
    ref: page 6 in https://arxiv.org/abs/1511.05440
    '''
    dgt1 = torch.abs(torch.diff(real,dim=-2))[:, :, 1:, 1:-1]
    dgt2 = torch.abs(torch.diff(real, dim=-1))[:, :, 1:-1, 1:]
    dpred1 = torch.abs(torch.diff(fake, dim=-2))[:, :, 1:, 1:-1]
    dpred2 = torch.abs(torch.diff(fake, dim=-1))[:, :, 1:-1, 1:]
    return 10*torch.log10(1.**2/torch.mean(torch.abs(dgt1+dgt2-dpred1-dpred2))).cpu().item()