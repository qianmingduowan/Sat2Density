import torch
from torch import nn
import functools
import torch.nn.functional as F

import sys 
sys.path.append(".") 
from model import geometry_transform
from imaginaire.utils.distributed import master_only_print as print
from model.graphs.decoder import DeepLab
from imaginaire.generators.craft_base import *



class Generator(nn.Module):
    def __init__(self,opt):
        super(Generator, self).__init__()
        gen_cfg = opt.arch.gen
        data_cfg = opt.data
        # self.gen_model = gen_model
        self.gen_cfg = opt.arch.gen
        if gen_cfg.transform_mode in ['project_RGB','volum_rendering','proj_like_radus']:
            self.pano_direction = torch.from_numpy(geometry_transform.get_original_coord(opt)).unsqueeze(0).to(opt.device)
        if gen_cfg.transform_mode == 'volum_rendering':
            last_act = 'relu'
        else:
            last_act = 'softmax'
        self.depth_model = inner_Generator(gen_cfg,gen_cfg.depth_arch,data_cfg,num_input_channels=3,last_act=last_act)
        render_input_channel = 3
        if gen_cfg.cat_opa:
            render_input_channel = render_input_channel+1
        self.denoise_model = inner_Generator(gen_cfg,gen_cfg.render_arch,data_cfg,render_input_channel,last_act='sigmoid')

        self.PE = None



    def forward(self, inputs, style_img=None,opt=None):
        estimated_height = self.depth_model(inputs)

        if self.gen_cfg.transform_mode in ['project_RGB','volum_rendering','proj_like_radus']:
            geo_outputs = geometry_transform.render(opt,inputs,estimated_height,self.pano_direction,PE=self.PE)
            generator_inputs,opacity,depth = geo_outputs['rgb'],geo_outputs['opacity'],geo_outputs['depth']
            if 'voxel' in geo_outputs.keys():
                voxel = geo_outputs['voxel']
        # mu, logvar, z = self.style_encode(style_img)
        # z = self.style_model(z)
        if self.gen_cfg.cat_opa:
            generator_inputs = torch.cat((generator_inputs,opacity),dim=1)
        output_RGB = self.denoise_model(generator_inputs)
        out_put = {
            'pred': output_RGB,
            # 'inter_RGB': generator_inputs,  ### out_feature not for show
            # 'mu' :mu,
            # 'logvar' : logvar,
            }
        if self.gen_cfg.transform_mode in ['volum_rendering']:
            out_put['opacity'] = opacity
        if self.gen_cfg.transform_mode:
            out_put['estimated_height'] = estimated_height
        out_put['generator_inputs'] = generator_inputs
        out_put['voxel'] = voxel
        out_put['depth'] = depth
        return out_put

