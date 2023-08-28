import numpy as np
import torch
import torch.nn as nn
from torch.nn import Upsample as NearestUpsample
import torch.nn.functional as F
from functools import partial

import sys 
sys.path.append(".") 
from imaginaire.layers import Conv2dBlock, LinearBlock, Res2dBlock


class StyleMLP(nn.Module):
    r"""MLP converting style code to intermediate style representation."""

    def __init__(self, style_dim, out_dim, hidden_channels=256, leaky_relu=True, num_layers=5, normalize_input=True,
                 output_act=True):
        super(StyleMLP, self).__init__()

        self.normalize_input = normalize_input
        self.output_act = output_act
        fc_layers = []
        fc_layers.append(nn.Linear(style_dim, hidden_channels, bias=True))
        for i in range(num_layers-1):
            fc_layers.append(nn.Linear(hidden_channels, hidden_channels, bias=True))
        self.fc_layers = nn.ModuleList(fc_layers)

        self.fc_out = nn.Linear(hidden_channels, out_dim, bias=True)

        if leaky_relu:
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.act = partial(F.relu, inplace=True)

    def forward(self, z):
        r""" Forward network

        Args:
            z (N x style_dim tensor): Style codes.
        """
        if self.normalize_input:
            z = F.normalize(z, p=2, dim=-1,eps=1e-6)
        for fc_layer in self.fc_layers:
            z = self.act(fc_layer(z))
        z = self.fc_out(z)
        if self.output_act:
            z = self.act(z)
        return z

class histo_process(nn.Module):
    r"""Histo process to replace Style Encoder constructor.

    Args:
        style_enc_cfg (obj): Style encoder definition file.
    """
    def __init__(self,style_enc_cfg):
        super().__init__()
        # if style_enc_cfg.histo.mode in ['RGB','rgb']:
        input_channel=270
        # else:
            # input_channel=90
        style_dims = style_enc_cfg.style_dims
        self.no_vae = getattr(style_enc_cfg, 'no_vae', False)
        num_filters = getattr(style_enc_cfg, 'num_filters', 180)
        self.process_model = nn.ModuleList()
        self.layer1 = LinearBlock(input_channel,num_filters)
        self.layer4 = LinearBlock(num_filters, num_filters)
        self.fc_mu = LinearBlock(num_filters, style_dims,nonlinearity='tanh')
        if not self.no_vae:
            self.fc_var = LinearBlock(num_filters, style_dims,nonlinearity='tanh')


    def forward(self,histo):
        x = self.layer1(histo)
        x = self.layer4(x)
        mu = self.fc_mu(x) #[-1,1]
        if not self.no_vae:
            logvar = self.fc_var(x) # [-1,1]
            std = torch.exp(0.5 * logvar)  # [0.607,1.624]
            eps = torch.randn_like(std) 
            z = eps.mul(std) + mu
        else:
            z = mu
            logvar = torch.zeros_like(mu)
        return mu, logvar, z



class StyleEncoder(nn.Module):
    r"""Style Encoder constructor.

    Args:
        style_enc_cfg (obj): Style encoder definition file.
    """

    def __init__(self, style_enc_cfg):
        super(StyleEncoder, self).__init__()
        input_image_channels = style_enc_cfg.input_image_channels
        num_filters = style_enc_cfg.num_filters
        kernel_size = style_enc_cfg.kernel_size
        padding = int(np.ceil((kernel_size - 1.0) / 2))
        style_dims = style_enc_cfg.style_dims
        weight_norm_type = style_enc_cfg.weight_norm_type
        self.no_vae = getattr(style_enc_cfg, 'no_vae', False)
        activation_norm_type = 'none'
        nonlinearity = 'leakyrelu'
        base_conv2d_block = \
            partial(Conv2dBlock,
                              kernel_size=kernel_size,
                              stride=2,
                              padding=padding,
                              weight_norm_type=weight_norm_type,
                              activation_norm_type=activation_norm_type,
                              # inplace_nonlinearity=True,
                              nonlinearity=nonlinearity)
        self.layer1 = base_conv2d_block(input_image_channels, num_filters)
        self.layer2 = base_conv2d_block(num_filters * 1, num_filters * 2)
        self.layer3 = base_conv2d_block(num_filters * 2, num_filters * 4)
        self.layer4 = base_conv2d_block(num_filters * 4, num_filters * 8)
        self.layer5 = base_conv2d_block(num_filters * 8, num_filters * 8)
        self.layer6 = base_conv2d_block(num_filters * 8, num_filters * 8)
        self.fc_mu = LinearBlock(num_filters * 8 * 4 * 4, style_dims,nonlinearity='tanh')
        if not self.no_vae:
            self.fc_var = LinearBlock(num_filters * 8 * 4 * 4, style_dims,nonlinearity='tanh')

    def forward(self, input_x):
        r"""SPADE Style Encoder forward.

        Args:
            input_x (N x 3 x H x W tensor): input images.
        Returns:
            mu (N x C tensor): Mean vectors.
            logvar (N x C tensor): Log-variance vectors.
            z (N x C tensor): Style code vectors.
        """
        if input_x.size(2) != 256 or input_x.size(3) != 256:
            input_x = F.interpolate(input_x, size=(256, 256), mode='bilinear')
        x = self.layer1(input_x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        if not self.no_vae:
            logvar = self.fc_var(x)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = eps.mul(std) + mu
        else:
            z = mu
            logvar = torch.zeros_like(mu)
        return mu, logvar, z


class RenderCNN(nn.Module):
    r"""CNN converting intermediate feature map to final image."""

    def __init__(self, in_channels, style_dim, hidden_channels=256,
                 leaky_relu=True):
        super(RenderCNN, self).__init__()
        self.fc_z_cond = nn.Linear(style_dim, 2 * 2 * hidden_channels)

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, stride=1, padding=0)
        self.conv2a = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1, bias=False)

        self.conv3a = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1, bias=False)

        self.conv4a = nn.Conv2d(hidden_channels, hidden_channels, 1, stride=1, padding=0)
        self.conv4b = nn.Conv2d(hidden_channels, hidden_channels, 1, stride=1, padding=0)

        self.conv4 = nn.Conv2d(hidden_channels, 3, 1, stride=1, padding=0)

        if leaky_relu:
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.act = partial(F.relu, inplace=True)

    def modulate(self, x, w_, b_):
        w_ = w_[..., None, None]
        b_ = b_[..., None, None]
        return x * (w_+1) + b_ +1e-9

    def forward(self, x, z):
        r"""Forward network.

        Args:
            x (N x in_channels x H x W tensor): Intermediate feature map
            z (N x style_dim tensor): Style codes.
        """
        z = self.fc_z_cond(z)
        adapt = torch.chunk(z, 2 * 2, dim=-1)
        y = self.act(self.conv1(x))

        y = y + self.conv2b(self.act(self.conv2a(y)))
        y = self.act(self.modulate(y, adapt[0], adapt[1]))

        y = y + self.conv3b(self.act(self.conv3a(y)))
        y = self.act(self.modulate(y, adapt[2], adapt[3]))

        y = y + self.conv4b(self.act(self.conv4a(y)))
        y = self.act(y)

        y = self.conv4(y)
        y = torch.sigmoid(y)
        return y


class inner_Generator(nn.Module):
    r"""Pix2pixHD coarse-to-fine generator constructor.

    Args:
        gen_cfg (obj): Generator definition part of the yaml config file.
        data_cfg (obj): Data definition part of the yaml config file.
        last_act:  ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``,default is 'relu'.
    """

    def __init__(self, gen_cfg,inner_cfg, data_cfg,num_input_channels=3,last_act='relu'):
        super().__init__()
        assert last_act in ['none', 'relu', 'leakyrelu', 'prelu',
            'tanh' , 'sigmoid' , 'softmax']
        # pix2pixHD has a global generator.
        global_gen_cfg = inner_cfg
        # By default, pix2pixHD using instance normalization.
        activation_norm_type = getattr(gen_cfg, 'activation_norm_type',
                                       'instance')
        activation_norm_params = getattr(gen_cfg, 'activation_norm_params',
                                         None)
        weight_norm_type = getattr(gen_cfg, 'weight_norm_type', '')
        padding_mode = getattr(gen_cfg, 'padding_mode', 'reflect')
        base_conv_block = partial(Conv2dBlock,
                                  padding_mode=padding_mode,
                                  weight_norm_type=weight_norm_type,
                                  activation_norm_type=activation_norm_type,
                                  activation_norm_params=activation_norm_params,
                                  nonlinearity='relu')
        base_res_block = partial(Res2dBlock,
                                 padding_mode=padding_mode,
                                 weight_norm_type=weight_norm_type,
                                 activation_norm_type=activation_norm_type,
                                 activation_norm_params=activation_norm_params,
                                 nonlinearity='relu', order='CNACN')
        # Know what is the number of available segmentation labels.

        # Global generator model.
        global_model = GlobalGenerator(global_gen_cfg, data_cfg,
                                       num_input_channels, padding_mode,
                                       base_conv_block, base_res_block,last_act=last_act)
        self.global_model = global_model


    def forward(self, input, random_style=False):
        r"""Coarse-to-fine generator forward.

        Args:
            data (dict) : Dictionary of input data.
            random_style (bool): Always set to false for the pix2pixHD model.
        Returns:
            output (dict) : Dictionary of output data.
        """
        return self.global_model(input)



    def load_pretrained_network(self, pretrained_dict):
        r"""Load a pretrained network."""
        # print(pretrained_dict.keys())
        model_dict = self.state_dict()
        print('Pretrained network has fewer layers; The following are '
              'not initialized:')

        not_initialized = set()
        for k, v in model_dict.items():
            kp = 'module.' + k.replace('global_model.', 'global_model.model.')
            if kp in pretrained_dict and v.size() == pretrained_dict[kp].size():
                model_dict[k] = pretrained_dict[kp]
            else:
                not_initialized.add('.'.join(k.split('.')[:2]))
        print(sorted(not_initialized))
        self.load_state_dict(model_dict)

    def inference(self, data, **kwargs):
        r"""Generator inference.

        Args:
            data (dict) : Dictionary of input data.
        Returns:
            fake_images (tensor): Output fake images.
            file_names (str): Data file name.
        """
        output = self.forward(data, **kwargs)
        return output['fake_images'], data['key']['seg_maps'][0]


class GlobalGenerator(nn.Module):
    r"""Coarse generator constructor. This is the main generator in the
    pix2pixHD architecture.

    Args:
        gen_cfg (obj): Generator definition part of the yaml config file.
        data_cfg (obj): Data definition part of the yaml config file.
        num_input_channels (int): Number of segmentation labels.
        padding_mode (str): zero | reflect | ...
        base_conv_block (obj): Conv block with preset attributes.
        base_res_block (obj): Residual block with preset attributes.
        last_act (str, optional, default='relu'):
            Type of nonlinear activation function.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
    """

    def __init__(self, gen_cfg, data_cfg, num_input_channels, padding_mode,
                 base_conv_block, base_res_block,last_act='relu'):
        super(GlobalGenerator, self).__init__()

        # num_img_channels = get_paired_input_image_channel_number(data_cfg)
        num_out_put_channels = getattr(gen_cfg, 'output_nc', 64)
        num_filters = getattr(gen_cfg, 'num_filters', 64)
        num_downsamples = getattr(gen_cfg, 'num_downsamples', 4)
        num_res_blocks = getattr(gen_cfg, 'num_res_blocks', 9)
        # First layer.
        model = [base_conv_block(num_input_channels, num_filters,
                                 kernel_size=7, padding=3)]
        # Downsample.
        for i in range(num_downsamples):
            ch = num_filters * (2 ** i)
            model += [base_conv_block(ch, ch * 2, 3, padding=1, stride=2)]
        # ResNet blocks.
        ch = num_filters * (2 ** num_downsamples)
        for i in range(num_res_blocks):
            model += [base_res_block(ch, ch, 3, padding=1)]
        # Upsample.
        num_upsamples = num_downsamples
        for i in reversed(range(num_upsamples)):
            ch = num_filters * (2 ** i)
            model += \
                [NearestUpsample(scale_factor=2),
                 base_conv_block(ch * 2, ch, 3, padding=1)]
        model += [Conv2dBlock(num_filters, num_out_put_channels, 7, padding=3,
                              padding_mode=padding_mode, nonlinearity=last_act)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        r"""Coarse-to-fine generator forward.

        Args:
            input (4D tensor) : Input semantic representations.
        Returns:
            output (4D tensor) : Synthesized image by generator.
        """
        return self.model(input)

class inner_Generator_split(nn.Module):
    r"""Pix2pixHD coarse-to-fine generator constructor.

    Args:
        gen_cfg (obj): Generator definition part of the yaml config file.
        data_cfg (obj): Data definition part of the yaml config file.
        last_act:  ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``,default is 'relu'.
    """

    def __init__(self, gen_cfg,inner_cfg, data_cfg,num_input_channels=3,last_act='relu'):
        super().__init__()
        assert last_act in ['none', 'relu', 'leakyrelu', 'prelu',
            'tanh' , 'sigmoid' , 'softmax']
        # pix2pixHD has a global generator.
        # By default, pix2pixHD using instance normalization.
        print(inner_cfg)
        style_dim =  gen_cfg.style_enc_cfg.interm_style_dims
        activation_norm_type = getattr(gen_cfg, 'activation_norm_type',
                                       'instance')
        activation_norm_params = getattr(gen_cfg, 'activation_norm_params',
                                         None)
        weight_norm_type = getattr(gen_cfg, 'weight_norm_type', '')
        padding_mode = getattr(gen_cfg, 'padding_mode', 'reflect')
        # num_input_channels = get_paired_input_label_channel_number(data_cfg)
        # num_input_channels = 3
        base_conv_block = partial(Conv2dBlock,
                                  padding_mode=padding_mode,
                                  weight_norm_type=weight_norm_type,
                                  activation_norm_type=activation_norm_type,
                                  activation_norm_params=activation_norm_params,
                                )
        base_res_block = partial(Res2dBlock,
                                 padding_mode=padding_mode,
                                 weight_norm_type=weight_norm_type,
                                 activation_norm_type=activation_norm_type,
                                 activation_norm_params=activation_norm_params,
                                 nonlinearity='relu', order='CNACN')
        # Know what is the number of available segmentation labels.

        # Global generator model.

        num_out_put_channels = getattr(inner_cfg, 'output_nc', 64)
        num_filters = getattr(inner_cfg, 'num_filters', 64)
        num_downsamples = 4
        num_res_blocks = getattr(inner_cfg, 'num_res_blocks', 9)
        # First layer.
        model = [base_conv_block(num_input_channels, num_filters,
                                 kernel_size=7, padding=3)]
        model += [nn.PReLU()]
        # Downsample.
        for i in range(num_downsamples):
            ch = num_filters * (2 ** i)
            model += [base_conv_block(ch, ch * 2, 3, padding=1, stride=2)]
            model += [nn.PReLU()]
        # ResNet blocks.
        ch = num_filters * (2 ** num_downsamples)
        for i in range(num_res_blocks):
            model += [base_res_block(ch, ch, 3, padding=1)]
        self.model = nn.Sequential(*model)
        # Upsample.
        assert num_downsamples == 4
        if not (inner_cfg.name =='render' and gen_cfg.style_inject):
            list = [16,8,4,2]
        else:
            list = [16,6,6,6]

        self.up0_a = NearestUpsample(scale_factor=2)
        self.up0_b = base_conv_block(num_filters * list[0], num_filters*list[1], 3, padding=1)
        self.up1_a = NearestUpsample(scale_factor=2)
        self.up1_b = base_conv_block(num_filters * list[1], num_filters*list[2], 3, padding=1)
        self.up2_a = NearestUpsample(scale_factor=2)
        self.up2_b = base_conv_block(num_filters * list[2], num_filters*list[3], 3, padding=1)
        self.up3_a = NearestUpsample(scale_factor=2)
        self.up3_b = base_conv_block(num_filters * list[3], num_filters, 3, padding=1)
        self.up_end = Conv2dBlock(num_filters, num_out_put_channels, 7, padding=3,
                              padding_mode=padding_mode, nonlinearity=last_act)
        if inner_cfg.name =='render' and gen_cfg.style_inject:
            self.fc_z_cond = nn.Linear(style_dim, 4* list[-1] * num_filters)

    def modulate(self, x, w, b):
        w = w[..., None, None]
        b = b[..., None, None]
        return x * (w+1) + b

    def forward(self, input,z=None):
        r"""Coarse-to-fine generator forward.

        Args:
            input (4D tensor) : Input semantic representations.
        Returns:
            output (4D tensor) : Synthesized image by generator.
        """
        if z is not None:
            z = self.fc_z_cond(z)
            adapt = torch.chunk(z, 2 * 2, dim=-1)
        input = self.model(input)
        input = self.up0_a(input)
        input = self.up0_b(input)
        input = F.leaky_relu(input,negative_slope=0.2, inplace=True)
        input = self.up1_a(input)
        input = self.up1_b(input)
        if z is not None:
            input = self.modulate(input, adapt[0], adapt[1])
        input = F.leaky_relu(input,negative_slope=0.2, inplace=True)

        input = self.up2_a(input)
        input = self.up2_b(input)
        if z is not None:
            input = self.modulate(input, adapt[2], adapt[3])
        input = F.leaky_relu(input,negative_slope=0.2, inplace=True)

        input = self.up3_a(input)
        input = self.up3_b(input)
        input = F.leaky_relu(input,negative_slope=0.2, inplace=True)

        input = self.up_end(input)

        return input

if __name__=='__main__':
    from easydict import EasyDict as edict
    style_enc_cfg = edict()
    style_enc_cfg.histo.mode = 'RGB'
    style_enc_cfg.histo.num_filters = 180
    model = histo_process(style_enc_cfg)