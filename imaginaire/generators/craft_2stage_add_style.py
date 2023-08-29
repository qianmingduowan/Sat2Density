import torch
from torch import nn
import sys 
sys.path.append(".") 
from model import geometry_transform
from imaginaire.generators.craft_base import *



class Generator(nn.Module):
    def __init__(self,opt):
        super(Generator, self).__init__()
        gen_cfg = opt.arch.gen
        data_cfg = opt.data
        style_enc_cfg = opt.arch.gen.style_enc_cfg
        # self.gen_model = gen_model
        self.style_inject = getattr(gen_cfg, 'style_inject',
                                       None)
        self.gen_cfg = opt.arch.gen
        self.pano_direction = torch.from_numpy(geometry_transform.get_original_coord(opt)).unsqueeze(0).to(opt.device)
        last_act = 'relu'
        self.depth_model = inner_Generator_split(gen_cfg,gen_cfg.depth_arch,data_cfg,num_input_channels=3,last_act=last_act)


        render_input_channel = 3
        if gen_cfg.cat_opa:
            render_input_channel +=1
        if gen_cfg.cat_depth:
            render_input_channel +=1

        self.denoise_model = inner_Generator_split(gen_cfg,gen_cfg.render_arch,data_cfg,render_input_channel,last_act='sigmoid')
        if self.style_inject:
            if self.style_inject=='histo':
                self.style_encode = histo_process(style_enc_cfg)
            elif self.style_inject=='perspective':
                self.style_encode = StyleEncoder(style_enc_cfg)
            else:
                raise Exception('Unknown style inject')
            self.style_model = StyleMLP(style_dim=style_enc_cfg.style_dims, out_dim=style_enc_cfg.interm_style_dims, hidden_channels=style_enc_cfg.hidden_channel, leaky_relu=True, num_layers=5, normalize_input=True,
                        output_act=True)

        self.PE = geometry_transform.position_produce(opt) if gen_cfg.cat_PE else None



    def forward(self, inputs, style_img=None,opt=None):
        # predicted height of satellite images
        estimated_height = self.depth_model(inputs)
        geo_outputs = geometry_transform.render(opt,inputs,estimated_height,self.pano_direction,PE=self.PE)
        generator_inputs,opacity,depth = geo_outputs['rgb'],geo_outputs['opacity'],geo_outputs['depth']
        if 'voxel' in geo_outputs.keys():
            voxel = geo_outputs['voxel']
                
        if self.gen_cfg.cat_opa:
            generator_inputs = torch.cat((generator_inputs,opacity),dim=1)
        if self.gen_cfg.cat_depth:
            generator_inputs = torch.cat((generator_inputs,depth),dim=1)
        if self.style_inject:
            mu, logvar, z = self.style_encode(style_img)
            z = self.style_model(z)
        else:
            z = None
        # merge multiple sources(rgb,opacity,depth and sky) and denoise redundancy
        output_RGB = self.denoise_model(generator_inputs,z)
        out_put = {'pred': output_RGB}
        if self.style_inject:
            out_put['mu'] = mu
            out_put['logvar']  = logvar
        out_put['estimated_height'] = estimated_height
        out_put['generator_inputs'] = generator_inputs
        out_put['voxel'] = voxel
        out_put['depth'] = depth
        out_put['opacity'] = opacity
        return out_put

