import importlib
import os
import os.path as osp
import sys
import warnings

import torch

import options
from utils import log

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from matplotlib.widgets import Cursor
from PIL import Image
from scipy.interpolate import interp1d, splev, splprep
from torch.utils.data import default_convert,default_collate

from model.geometry_transform import render_sat
import cv2 

def get_checkpoint(opt):
    if opt.test_ckpt_path == '2u87bj8w':
        opt.test_ckpt_path = osp.join('wandb/run-20230219_141512-2u87bj8w/files/checkpoint/model.pth')
    elif opt.test_ckpt_path == '2cqv8uh4':
        opt.test_ckpt_path = osp.join('wandb/run-20230303_142752-2cqv8uh4/files/checkpoint/model.pth')
    else:
        pass


def img_read(img,size=None,datatype='RGB'):
    img = Image.open(img).convert('RGB' if datatype=='RGB' else "L")
    if size:
        if type(size) is int:
            size = (size,size)
        img = img.resize(size = size,resample=Image.BICUBIC if datatype=='RGB' else Image.NEAREST)
    img = transforms.ToTensor()(img)
    return img

def select_points(sat_image):
    fig = plt.figure()
    fig.set_size_inches(1,1,forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.imshow(sat_image)

    coords = []

    def ondrag(event):
        if event.button != 1:
            return
        x, y = int(event.xdata), int(event.ydata)
        coords.append((x, y))
        ax.plot([x], [y], 'o', color='red')
        fig.canvas.draw_idle()
        
    fig.add_axes(ax)
    cursor = Cursor(ax, useblit=True, color='red', linewidth=1)
    fig.canvas.mpl_connect('motion_notify_event', ondrag)
    plt.show()
    plt.close()

    unique_lst = list(dict.fromkeys(coords))
    pixels = []
    for x in coords:
        if x in unique_lst:
            if x not in pixels:
                pixels.append(x)
    print(pixels)
    pixels = np.array(pixels)
    tck, u = splprep(pixels.T, s=25, per=0)
    u_new = np.linspace(u.min(), u.max(), 80)
    x_new, y_new = splev(u_new, tck)

    smooth_path = np.array([x_new,y_new]).T
    return pixels, smooth_path

def volume2pyvista(volume_data):
    import pyvista as pv 
    grid = pv.UniformGrid()
    grid.dimensions = volume_data.shape
    grid.spacing = (1, 1, 1)
    grid.origin = (0, 0, 0)
    grid.point_data['values'] = volume_data.flatten(order='F')
    return grid

@torch.no_grad()
def test_vid(model, opt):
    ckpt = torch.load(opt.test_ckpt_path, map_location='cpu')
    model.netG.load_state_dict(ckpt['netG'])
    model.netG.eval()
    
    # for idx, data in enumerate(model.val_loader):
    #     import pdb; pdb.set_trace()
    demo_imgpath = opt.demo_img 
    sty_imgpath = opt.sty_img 
    if opt.sky_img is None:
        sky_imgpath = opt.sty_img.replace('image','sky')
    else:
        sky_imgpath = opt.sky_img

    sat = img_read(demo_imgpath, size=opt.data.sat_size)
    pano = img_read(sty_imgpath, size=opt.data.pano_size)

    input_dict = {}
    input_dict['sat'] = sat
    input_dict['pano'] = pano
    input_dict['paths'] = demo_imgpath


    if opt.data.sky_mask:
        sky = img_read(sky_imgpath, size=opt.data.pano_size, datatype='L') 
        black_ground = torch.zeros_like(pano)
        input_a = (pano*sky+black_ground*(1-sky))
        if opt.data.histo_mode == 'grey':
            sky_histc = input_a.histc()[10:]
        elif opt.data.histo_mode in ['rgb','RGB']:
            sky_histc = [torch.histc(input_a[i])[10:] for i in range(3)]
            sky_histc = torch.cat(sky_histc)
        input_dict['sky_histc'] = sky_histc
        input_dict['sky_mask'] = sky
    else:
        sky_histc = None
    
    for key in input_dict.keys():
        if isinstance(input_dict[key], torch.Tensor):
            input_dict[key] = input_dict[key].unsqueeze(0)

    model.set_input(input_dict)
    
    model.style_temp = model.sky_histc
    
    pixels, smooth_path = select_points(sat_image=sat.permute(1,2,0).numpy())

    rendered_image_list = []
    rendered_depth_list = []

    volume_data = None

    for i, (x,y) in enumerate(pixels):
        opt.origin_H_W = [(y-128)/128, (x-128)/128] # TODO: hard code should be removed in the future
        print('Rendering at ({}, {})'.format(x,y))
        model.forward(opt)

        rgb = model.out_put.pred[0].clamp(min=0,max=1.0).cpu().numpy().transpose((1,2,0))
        rgb = np.array(rgb*255, dtype=np.uint8)
        rendered_image_list.append(rgb)

        rendered_depth_list.append(
            model.out_put.depth[0,0].cpu().numpy()
        )

    sat_opacity, sat_depth = render_sat(opt,model.out_put.voxel)
    
    volume_data = model.out_put.voxel[0].cpu().numpy().transpose((1,2,0))
    volume_data = np.clip(volume_data, None, 10)
    
    volume_export = volume2pyvista(volume_data)

    os.makedirs(opt.save_dir, exist_ok=True)
    volume_export.save(os.path.join(opt.save_dir, 'volume.vtk'))

    # save rendered images 
    os.makedirs(osp.join(opt.save_dir,'rendered_images'), exist_ok=True)
    for i, img in enumerate(rendered_image_list):
        img = np.clip(img, 0, 1)
        plt.imsave(osp.join(opt.save_dir,'rendered_images','{:05d}.png'.format(i)), img)

    os.makedirs(osp.join(opt.save_dir,'rendered_depth'), exist_ok=True)

    os.makedirs(osp.join(opt.save_dir,
    'rendered_images+depths'), exist_ok=True)

    for i, img in enumerate(rendered_depth_list):
        depth = np.array(img/img.max()*255,dtype=np.uint8)
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_TURBO)
        plt.imsave(osp.join(opt.save_dir,'rendered_depth','{:05d}.png'.format(i)), depth)
        image_and_depth = np.concatenate((rendered_image_list[i], depth), axis=0)

        plt.imsave(osp.join(opt.save_dir,'rendered_images+depths','{:05d}.png'.format(i)), image_and_depth)
        
    print('Done')


def main():
    log.process(os.getpid())
    log.title("[{}] (PyTorch code for testing Sat2Density and debug".format(sys.argv[0]))
    
    opt_cmd = options.parse_arguments(sys.argv[1:])
    opt = options.set(opt_cmd=opt_cmd)
    opt.isTrain = False
    opt.name = opt.yaml if opt.name is None else opt.name
    opt.batch_size = 1

    if opt.save_dir is None:
        raise Exception("Please specify the save dir")

    get_checkpoint(opt)

    mode = importlib.import_module("model.{}".format(opt.model))
    m = mode.Model(opt)

    # m.load_dataset(opt)
    m.build_networks(opt)

    test_vid(m, opt)
    
    # import pdb; pdb.set_trace()
    
    # print(m)
    # # test or visualization
    # if opt.task == 'test_vid':
    #     m.test_vid(opt)
    # elif opt.task == 'test_sty':
    #     m.test_sty(opt)
    # elif opt.task == 'test_interpolation':
    #     m.test_interpolation(opt)
    # else:
    #     raise RuntimeError("Unknow task")

if __name__ == "__main__":
    main()