import numpy as np
import torch,math
from PIL import Image
import torchvision
from easydict import EasyDict as edict

def position_produce(opt): 
    depth_channel =  opt.arch.gen.depth_arch.output_nc 
    if  opt.optim.ground_prior:
        depth_channel = depth_channel+1
    z_ = torch.arange(depth_channel)/depth_channel
    x_ = torch.arange(opt.data.sat_size[1])/opt.data.sat_size[1]
    y_ = torch.arange(opt.data.sat_size[0])/opt.data.sat_size[0]
    Z,X,Y = torch.meshgrid(z_,x_,y_)
    input = torch.cat((Z[...,None],X[...,None],Y[...,None]),dim=-1).to(opt.device)
    pos = positional_encoding(opt,input)
    pos = pos.permute(3,0,1,2)
    return  pos

def positional_encoding(opt,input): # [B,...,N]
    shape = input.shape
    freq = 2**torch.arange(opt.arch.gen.PE_channel,dtype=torch.float32,device=opt.device)*np.pi # [L]
    spectrum = input[...,None]*freq # [B,...,N,L]
    sin,cos = spectrum.sin(),spectrum.cos() # [B,...,N,L]
    input_enc = torch.stack([sin,cos],dim=-2) # [B,...,N,2,L]
    input_enc = input_enc.view(*shape[:-1],-1) # [B,...,2NL]
    return input_enc



def get_original_coord(opt):
    '''
    pano_direction [X,Y,Z] x right,y up,z out
    '''
    W,H  = opt.data.pano_size
    _y = np.repeat(np.array(range(W)).reshape(1,W), H, axis=0)
    _x = np.repeat(np.array(range(H)).reshape(1,H), W, axis=0).T

    if opt.data.dataset in ['CVACT_Shi', 'CVACT', 'CVACThalf']:
        _theta = (1 - 2 * (_x) / H) * np.pi/2 # latitude 
    elif opt.data.dataset in ['CVUSA']:
        _theta = (1 - 2 * (_x) / H) * np.pi/4
    # _phi = math.pi* ( 1 -2* (_y)/W ) # longtitude 
    _phi = math.pi*( - 0.5 - 2* (_y)/W )
    axis0 = (np.cos(_theta)*np.cos(_phi)).reshape(H, W, 1)
    axis1 = np.sin(_theta).reshape(H, W, 1) 
    axis2 = (-np.cos(_theta)*np.sin(_phi)).reshape(H, W, 1) 
    pano_direction = np.concatenate((axis0, axis1, axis2), axis=2)
    return pano_direction  


def render(opt,feature,voxel,pano_direction,PE=None):
    '''
    render ground images from ssatellite images
    
    feature: B,C,H_sat,W_sat feature or a input RGB
    voxel: B,N,H_sat,W_sat density of each grid
    PE: whether add position encoding , default is None
    pano_direction: pano ray direction  by their definition
    '''
    # pano_W,pano_H = opt.data.pano_size
    sat_W,sat_H = opt.data.sat_size
    BS = feature.size(0)
    ##### get origin, sample point ,depth

    if opt.data.dataset =='CVACT_Shi':
        origin_height=2       ## the height of photo taken in real world scale
        realworld_scale = 30  ## the real world scale corresponding to [-1,1] regular cooridinate
    elif opt.data.dataset == 'CVUSA':
        origin_height=2       
        realworld_scale = 55  
    else:
        assert Exception('Not implement yet')

    assert sat_W==sat_H
    pixel_resolution = realworld_scale/sat_W #### pixel resolution of satellite image in realworld

    if opt.data.sample_total_length:
        sample_total_length = opt.data.sample_total_length
    else: sample_total_length = (int(max(np.sqrt((realworld_scale/2)**2+(realworld_scale/2)**2+(2)**2), \
        np.sqrt((realworld_scale/2)**2+(realworld_scale/2)**2+(opt.data.max_height-origin_height)**2))/pixel_resolution))/(sat_W/2)

    origin_z = torch.ones([BS,1])*(-1+(origin_height/(realworld_scale/2))) ### -1 is the loweast position in regular cooridinate
    ##### origin_z: which can be definition by origin height
    if opt.origin_H_W is None: ### origin_H_W is the photo taken space in regular coordinate
        origin_H,origin_w = torch.zeros([BS,1]),torch.zeros([BS,1])   
    else:
        origin_H,origin_w = torch.ones([BS,1])*opt.origin_H_W[0],torch.ones([BS,1])*opt.origin_H_W[1]
    origin = torch.cat([origin_w,origin_z,origin_H],dim=1).to(opt.device)[:,None,None,:]  ## w,z,h, samiliar to NERF coordinate definition
    sample_len = ((torch.arange(opt.data.sample_number)+1)*(sample_total_length/opt.data.sample_number)).to(opt.device)
    ### sample_len:  For sample distance is fixed, so we can easily calculate sample len along a way by max length and sample number
    origin = origin[...,None]
    pano_direction = pano_direction[...,None] ### the direction has been normalized
    depth = sample_len[None,None,None,None,:]
    sample_point = origin + pano_direction * depth #0.0000],-0.8667],0.0000 w,z,h
    # x points right, y points up, z points backwards scene nerf
    # ray_depth = sample_point-origin

    if opt.optim.ground_prior:
        voxel = torch.cat([torch.ones(voxel.size(0),1,voxel.size(2),voxel.size(3),device=opt.device)*1000,voxel],1)

            # voxel[:,0,:,:] = 100
    N = voxel.size(1)
    voxel_low = -1
    voxel_max = -1 + opt.data.max_height/(realworld_scale/2)  ### voxel highest space in normal space
    grid = sample_point.permute(0,4,1,2,3)[...,[0,2,1]] ### BS,NUM_point,W,H,3 
    grid[...,2]   = ((grid[...,2]-voxel_low)/(voxel_max-voxel_low))*2-1  ### grid_space change to sample space by scale the z space
    grid = grid.float()  ## [1, 300, 256, 512, 3]
    
    color_input = feature.unsqueeze(2).repeat(1, 1, N, 1, 1)
    alpha_grid = torch.nn.functional.grid_sample(voxel.unsqueeze(1), grid)

    color_grid = torch.nn.functional.grid_sample(color_input, grid)
    if PE is not None:
        PE_grid = torch.nn.functional.grid_sample(PE[None,...], grid[:1,...])
        color_grid = torch.cat([color_grid,PE_grid.repeat(BS, 1, 1, 1, 1)],dim=1)

    depth_sample = depth.permute(0,1,2,4,3).view(1,-1,opt.data.sample_number,1)
    feature_size = color_grid.size(1)
    color_grid = color_grid.permute(0,3,4,2,1).view(BS,-1,opt.data.sample_number,feature_size)
    alpha_grid = alpha_grid.permute(0,3,4,2,1).view(BS,-1,opt.data.sample_number)
    intv = sample_total_length/opt.data.sample_number
    output = composite(opt, rgb_samples=color_grid,density_samples=alpha_grid,depth_samples=depth_sample,intv = intv)
    output['voxel']  = voxel
    return output

def composite(opt,rgb_samples,density_samples,depth_samples,intv):
    """generate 2d ground images according to ray

    Args:
        opt (_type_): option dict
        rgb_samples (_type_): rgb (sampled from satellite image) belongs to the ray which start from the ground camera to world
        density_samples (_type_): density (sampled from the predicted voxel of satellite image) belongs to the ray which start from the ground camera to world
        depth_samples (_type_): depth of the ray which start from the ground camera to world
        intv (_type_): interval of the ray's depth which start from the ground camera to world

    Returns:
        2d ground images (rgd, opacity, and depth)
    """    
    
    sigma_delta = density_samples*intv # [B,HW,N]
    alpha = 1-(-sigma_delta).exp_() # [B,HW,N]
    T = (-torch.cat([torch.zeros_like(sigma_delta[...,:1]),sigma_delta[...,:-1]],dim=2).cumsum(dim=2)) .exp_() # [B,HW,N]
    prob = (T*alpha)[...,None] # [B,HW,N,1]
    # integrate RGB and depth weighted by probability
    depth = (depth_samples*prob).sum(dim=2) # [B,HW,1]
    rgb = (rgb_samples*prob).sum(dim=2) # [B,HW,3]
    opacity = prob.sum(dim=2) # [B,HW,1]
    depth = depth.permute(0,2,1).view(depth.size(0),-1,opt.data.pano_size[1],opt.data.pano_size[0])
    rgb = rgb.permute(0,2,1).view(rgb.size(0),-1,opt.data.pano_size[1],opt.data.pano_size[0])
    opacity = opacity.view(opacity.size(0),1,opt.data.pano_size[1],opt.data.pano_size[0])
    return {'rgb':rgb,'opacity':opacity,'depth':depth}


def get_sat_ori(opt):
    W,H  = opt.data.sat_size
    y_range =  (torch.arange(H,dtype=torch.float32,)+0.5)/(0.5*H)-1
    x_range = (torch.arange(W,dtype=torch.float32,)+0.5)/(0.5*H)-1
    Y,X = torch.meshgrid(y_range,x_range)
    Z = torch.ones_like(Y)
    xy_grid = torch.stack([X,Z,Y],dim=-1)[None,:,:]
    return xy_grid

def render_sat(opt,voxel):
    '''
    voxel: voxel has been processed
    '''
    # pano_W,pano_H = opt.data.pano_size
    sat_W,sat_H = opt.data.sat_size
    sat_ori  = get_sat_ori(opt)
    sat_dir  = torch.tensor([0,-1,0])[None,None,None,:]

    ##### get origin, sample point ,depth
    if opt.data.dataset =='CVACT_Shi':
        origin_height=2      
        realworld_scale = 30  
    elif opt.data.dataset == 'CVUSA':
        origin_height=2       
        realworld_scale = 55  

    else:
        assert Exception('Not implement yet')

    pixel_resolution = realworld_scale/sat_W #### pixel resolution of satellite image in realworld
    # if opt.data.sample_total_length:
    #     sample_total_length = opt.data.sample_total_length
    # else: sample_total_length = (int(max(np.sqrt((realworld_scale/2)**2+(realworld_scale/2)**2+(2)**2), \
    #     np.sqrt((realworld_scale/2)**2+(realworld_scale/2)**2+(opt.data.max_height-origin_height)**2))/pixel_resolution))/(sat_W/2)
    sample_total_length = 2
    # #### sample_total_length: it can be definition in future, which is the farest length between sample point and original ponit 
    # assert sat_W==sat_H

    origin = sat_ori.to(opt.device)  ## w,z,h, samiliar to NERF coordinate definition
    sample_len = ((torch.arange(opt.data.sample_number)+1)*(sample_total_length/opt.data.sample_number)).to(opt.device)
    ### sample_len:  For sample distance is fixed, so we can easily calculate sample len along a way by max length and sample number
    origin = origin[...,None].to(opt.device)
    direction = sat_dir[...,None].to(opt.device) ### the direction has been normalized
    depth = sample_len[None,None,None,None,:]
    sample_point = origin + direction * depth #0.0000],-0.8667],0.0000 w,z,h


    N = voxel.size(1)
    voxel_low = -1
    voxel_max = -1 + opt.data.max_height/(realworld_scale/2)  ### voxel highest space in normal space
    # axis_voxel = (torch.arange(N)/N) * (voxel_max-voxel_low) +voxel_low
    grid = sample_point.permute(0,4,1,2,3)[...,[0,2,1]] ### BS,NUM_point,W,H,3 
    grid[...,2]   = ((grid[...,2]-voxel_low)/(voxel_max-voxel_low))*2-1  ### grid_space change to sample space by scale the z space
    grid = grid.float()  ## [1, 300, 256, 512, 3]
    alpha_grid = torch.nn.functional.grid_sample(voxel.unsqueeze(1), grid)

    depth_sample = depth.permute(0,1,2,4,3).view(1,-1,opt.data.sample_number,1)
    alpha_grid = alpha_grid.permute(0,3,4,2,1).view(opt.batch_size,-1,opt.data.sample_number)
    # color_grid = torch.flip(color_grid,[2])
    # alpha_grid = torch.flip(alpha_grid,[2])
    intv = sample_total_length/opt.data.sample_number
    output = composite_sat(opt,density_samples=alpha_grid,depth_samples=depth_sample,intv = intv)
    return output['opacity'],output['depth']

def composite_sat(opt,density_samples,depth_samples,intv):
    sigma_delta = density_samples*intv # [B,HW,N]
    alpha = 1-(-sigma_delta).exp_() # [B,HW,N]
    T = (-torch.cat([torch.zeros_like(sigma_delta[...,:1]),sigma_delta[...,:-1]],dim=2).cumsum(dim=2)) .exp_() # [B,HW,N]
    prob = (T*alpha)[...,None] # [B,HW,N,1]
    depth = (depth_samples*prob).sum(dim=2) # [B,HW,1]
    opacity = prob.sum(dim=2) # [B,HW,1]
    depth = depth.permute(0,2,1).view(depth.size(0),-1,opt.data.sat_size[1],opt.data.sat_size[0])
    opacity = opacity.view(opacity.size(0),1,opt.data.sat_size[1],opt.data.sat_size[0])
    # return rgb,depth,opacity,prob # [B,HW,K]
    return {'opacity':opacity,'depth':depth}

if __name__ == '__main__':
    # test_demo
    opt=edict()
    opt.device = 'cuda'
    opt.data = edict()
    opt.data.pano_size = [512,256]
    opt.data.sat_size = [256,256]
    opt.data.dataset = 'CVACT_Shi'
    opt.data.max_height = 20
    opt.data.sample_number = 300
    opt.arch = edict()
    opt.optim = edict()
    opt.optim.ground_prior = False
    opt.arch.gen.transform_mode = 'volum_rendering'
    # opt.arch.gen.transform_mode = 'proj_like_radus'
    BS = 1
    opt.data.sample_total_length = 1
    sat_name = './CVACT/satview_correct/__-DFIFxvZBCn1873qkqXA_satView_polish.png'
    a = Image.open(sat_name)
    a = np.array(a).astype(np.float32)
    a = torch.from_numpy(a)
    a = a.permute(2, 0, 1).unsqueeze(0).to(opt.device).repeat(BS,1,1,1)/255.


    pano = sat_name.replace('satview_correct','streetview').replace('_satView_polish','_grdView')
    pano = np.array(Image.open(pano)).astype(np.float32)
    pano = torch.from_numpy(pano)
    pano = pano.permute(2, 0, 1).unsqueeze(0).to(opt.device).repeat(BS,1,1,1)/255.
    voxel=torch.zeros([BS, 65, 256, 256]).to(opt.device)
    pano_direction = torch.from_numpy(get_original_coord(opt)).unsqueeze(0).to(opt.device)

    import time
    star = time.time()
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
        rgb,opacity =render(opt,a,voxel,pano_direction)
    print(prof.table())
      
    print(time.time()-star) 

    torchvision.utils.save_image(torch.cat([rgb,pano],2), opt.arch.gen.transform_mode + '.png')
    print( opt.arch.gen.transform_mode + '.png')
    torchvision.utils.save_image(opacity, 'opa.png')