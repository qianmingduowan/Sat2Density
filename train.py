import os,sys
import importlib
import options
import warnings
warnings.filterwarnings("ignore")
os.environ['WANDB_IGNORE_GLOBS'] = '*.pth'  # not save checkpoint in cloud

def main():
    opt_cmd = options.parse_arguments(sys.argv[1:])
    opt = options.set(opt_cmd=opt_cmd)
    assert opt.task in ["train","Train"]
    opt.isTrain = True
    opt.name = opt.yaml if opt.name is None else opt.name
    mode = importlib.import_module("model.{}".format(opt.model))
    m = mode.Model(opt)

    m.load_dataset(opt)
    m.build_networks(opt)
    m.setup_optimizer(opt)
    m.train(opt)
    
if __name__=="__main__":
    main()
