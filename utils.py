import termcolor,os,shutil,torch
from easydict import EasyDict as edict
from collections import OrderedDict
import math
import numpy as np
from torch.nn import init

def get_time(sec):
    """
    Convert seconds to days, hours, minutes, and seconds
    """
    d = int(sec//(24*60*60))
    h = int(sec//(60*60)%24)
    m = int((sec//60)%60)
    s = int(sec%60)
    return d,h,m,s

# convert to colored strings
def red(message,**kwargs): return termcolor.colored(str(message),color="red",attrs=[k for k,v in kwargs.items() if v is True])
def green(message,**kwargs): return termcolor.colored(str(message),color="green",attrs=[k for k,v in kwargs.items() if v is True])
def blue(message,**kwargs): return termcolor.colored(str(message),color="blue",attrs=[k for k,v in kwargs.items() if v is True])
def cyan(message,**kwargs): return termcolor.colored(str(message),color="cyan",attrs=[k for k,v in kwargs.items() if v is True])
def yellow(message,**kwargs): return termcolor.colored(str(message),color="yellow",attrs=[k for k,v in kwargs.items() if v is True])
def magenta(message,**kwargs): return termcolor.colored(str(message),color="magenta",attrs=[k for k,v in kwargs.items() if v is True])
def grey(message,**kwargs): return termcolor.colored(str(message),color="grey",attrs=[k for k,v in kwargs.items() if v is True])



def openreadtxt(file_name):
    
    file = open(file_name,'r')  
    file_data = file.read().splitlines() 
    return file_data

def to_dict(D,dict_type=dict):
    D = dict_type(D)
    for k,v in D.items():
        if isinstance(v,dict):
            D[k] = to_dict(v,dict_type)
    return D

class Log:
    def __init__(self): pass
    def process(self,pid):
        print(grey("Process ID: {}".format(pid),bold=True))
    def title(self,message):
        print(yellow(message,bold=True,underline=True))
    def info(self,message):
        print(magenta(message,bold=True))
    def options(self,opt,level=0):
        for key,value in sorted(opt.items()):
            if isinstance(value,(dict,edict)):
                print("   "*level+cyan("* ")+green(key)+":")
                self.options(value,level+1)
            else:
                print("   "*level+cyan("* ")+green(key)+":",yellow(value))
    def loss_train(self,opt,ep,lr,loss,timer):
        if not opt.max_epoch: return
        message = grey("[train] ",bold=True)
        message += "epoch {}/{}".format(cyan(ep,bold=True),opt.max_epoch)
        message += ", lr:{}".format(yellow("{:.2e}".format(lr),bold=True))
        message += ", loss:{}".format(red("{:.3e}".format(loss),bold=True))
        message += ", time:{}".format(blue("{0}-{1:02d}:{2:02d}:{3:02d}".format(*get_time(timer.elapsed)),bold=True))
        message += " (ETA:{})".format(blue("{0}-{1:02d}:{2:02d}:{3:02d}".format(*get_time(timer.arrival))))
        print(message)
    def loss_val(self,opt,loss):
        message = grey("[val] ",bold=True)
        message += "loss:{}".format(red("{:.3e}".format(loss),bold=True))
        print(message)
log = Log()

def save_checkpoint(model,ep,latest=False,children=None,output_path=None):

    os.makedirs("{0}/model".format(output_path),exist_ok=True)
    checkpoint = dict(
        epoch=ep,
        netG=model.netG.state_dict(),
        netD=model.netD.state_dict()
        )

    torch.save(checkpoint,"{0}/model.pth".format(output_path))
    if not latest:
        shutil.copy("{0}/model.pth".format(output_path),
                    "{0}/model/{1}.pth".format(output_path,ep)) # if ep is None, track it instead

def filt_ckpt_keys(ckpt, item_name, model_name):
    # if item_name in ckpt:
    assert item_name in ckpt, "Cannot find [%s] in the checkpoints." % item_name
    d = ckpt[item_name]
    d_filt = OrderedDict()
    for k, v in d.items():
        k_list = k.split('.')
        if k_list[0] == model_name:
            if k_list[1] == 'module':
                d_filt['.'.join(k_list[2:])] = v
            else:
                d_filt['.'.join(k_list[1:])] = v
    return d_filt

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def filt_ckpt_keys(ckpt, item_name, model_name):
    # if item_name in ckpt:
    assert item_name in ckpt, "Cannot find [%s] in the checkpoints." % item_name
    d = ckpt[item_name]
    d_filt = OrderedDict()
    for k, v in d.items():
        k_list = k.split('.')
        if k_list[0] == model_name:
            if k_list[1] == 'module':
                d_filt['.'.join(k_list[2:])] = v
            else:
                d_filt['.'.join(k_list[1:])] = v
    return d_filt

def get_ray_pano(batch_img):
    _,_,H,W = batch_img.size()
    _y = np.repeat(np.array(range(W)).reshape(1,W), H, axis=0)
    _x = np.repeat(np.array(range(H)).reshape(1,H), W, axis=0).T
    
    _theta = (1 - 2 * (_x) / H) * np.pi/2 # latitude
    _phi = 2*math.pi*(0.5 - (_y)/W ) # longtitude
    axis0 = (np.cos(_theta)*np.cos(_phi)).reshape(1,H, W)
    axis1 = np.sin(_theta).reshape(1,H, W)
    axis2 = (-np.cos(_theta)*np.sin(_phi)).reshape(1, H, W)
    original_coord = np.concatenate((axis0, axis1, axis2), axis=0)

    return original_coord

def init_weights(net, init_type='kaiming', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


if __name__=='__main__':
    a = torch.zeros([2,3,200,100])
    cood = get_ray_pano(a)
