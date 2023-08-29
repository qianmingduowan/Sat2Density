import os,sys
import importlib
import options
from utils import log
import warnings
warnings.filterwarnings("ignore")
os.environ['WANDB_IGNORE_GLOBS'] = '*.pth'
os.environ['WANDB_MODE'] = 'dryrun'

def main():
    log.process(os.getpid())
    log.title("[{}] (PyTorch code for testing Sat2Density and debug".format(sys.argv[0]))
    opt_cmd = options.parse_arguments(sys.argv[1:])
    opt = options.set(opt_cmd=opt_cmd)
    if opt.test_ckpt_path and opt.task not in ["test" , "val","vis_test",'test_speed','test_vid','test_sty','test_interpolation']:
        opt.task = "test"
    if opt.task in ["train" , "Train"]:
        opt.isTrain = True
    else:
        opt.isTrain = False

    opt.name = opt.yaml if opt.name is None else opt.name
    mode = importlib.import_module("model.{}".format(opt.model))
    m = mode.Model(opt)

    m.load_dataset(opt)
    m.build_networks(opt)
    # train
    if opt.task in ["train" , "Train"]:
        m.setup_optimizer(opt)
        m.train(opt)

    # test or visualization
    elif opt.task in ["test" , "val","vis_test"]:
        m.test(opt)

    # test speed
    elif opt.task == 'test_speed':
        m.test_speed(opt)
    # inference video results
    elif opt.task == 'test_vid':
        m.test_vid(opt)
    # test one image with different styles
    elif opt.task == 'test_sty':
        m.test_sty(opt)
    # test style interpolation
    elif opt.task == 'test_interpolation':
        m.test_interpolation(opt)    
    else:
        raise Exception("Unknow task")



        

if __name__=="__main__":
    main()
