# Sat2Density: Faithful Density Learning from Satellite-Ground Image Pairs
## ICCV 2023

###
[Project](https://sat2density.github.io/) |
[Paper](https://arxiv.org/abs/2303.14672) 
<!-- [Video](https://www.youtube.com/watch?v=1Hky092CGFQ) | -->
<!-- [Two Minute Papers Video](https://www.youtube.com/watch?v=jl0XCslxwB0) -->



## Software Installation 
For installation, please check out [install.md](scripts/INSTALL.md).


## Hardware Requirement
We trained our model using 1 V100 32GB GPU. Training took about 20 hours.




## Pre-trained model
Then pre-trained model for CVACT & CVUSA   could be download from [CVACT](https://github.com/sat2density/checkpoints/releases/download/cvact/run-20230219_141512-2u87bj8w.zip) and [CVUSA](https://github.com/sat2density/checkpoints/releases/download/cvusa/run-20230303_142752-2cqv8uh4.zip) with wget seperately.
Please unzip and then put in in in `wandb` folder.

## Quick Start demo
synthesis video

`
bash inference/quick_demo_video.sh
`

style interpolation

`
bash inference/quick_demo_interpolation.sh
`

## Train & Inference
### data preparation
For data preparation, please check out [data.md](dataset/INSTALL.md).


### Inference

To test Center Ground-View Synthesis setting
If you want save results, please add --task=vis_test
```bash
# CVACT
python offline_train_test.py --yaml=sat2density_cvact --test_ckpt_path=2u87bj8w
# CVUSA
python offline_train_test.py --yaml=sat2density_cvusa --test_ckpt_path=2cqv8uh4
```

To test inference with different illumination
```bash
# CVACT
bash inference/single_style_test_cvact.sh
# CVUSA
bash inference/single_style_test_cvusa.sh
```

To test synthesis ground videos
```bash
bash inference/synthesis_video.sh
```

To test style interpolation
```bash 
#demo
python offline_train_test.py --task=test_interpolation --yaml=sat2density_cvact --test_ckpt_path=2u87bj8w 
--sty_img1=_jaGbjgbHAe78_VhcPtmZQ_grdView.png --sty_img2=pdZmLHYEhe2PHj_8-WHMhw_grdView.png --demo_img=VAMM6sIEbYAY5E6ZD_RMKg_satView_polish.png
```



## Training

### Training command

```bash
# CVACT
CUDA_VISIBLE_DEVICES=X python train.py --yaml=sat2density_cvact
# CVUSA
CUDA_VISIBLE_DEVICES=X python train.py --yaml=sat2density_cvusa
```

## Citation
If you use this code for your research, please cite

```
@inproceedings{qian2021sat2density,
  title={Sat2Density: Faithful Density Learning from Satellite-Ground Image Pairs},
  author={Qian, Ming and Xiong, Jincheng and Xia, Gui-Song and Xue, Nan},
  booktitle={ICCV},
  year={2023}
}
```

## License
This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License.
For commercial use, please contact [mingqian@whu.edu.cn].
