# Sat2Density: Faithful Density Learning from Satellite-Ground Image Pairs

> [Ming Qian](https://qianmingduowan.github.io/), Jincheng Xiong, [Gui-Song Xia](http://www.captain-whu.com/xia_En.html), [Nan Xue](https://xuenan.net)
>
> IEEE/CVF International Conference on Computer Vision (ICCV), 2023
>
> [Project Page](https://sat2density.github.io/) | [Paper](https://arxiv.org/abs/2303.14672) | [Data](dataset/INSTALL.md) | [Install.md](scripts/INSTALL.md)

> <p align="center" float="left">
>    <img src="docs/figures/demo/case1.sat.gif" alt="drawing" width="19%">  
>    <img src="docs/figures/demo-density/case1.gif" alt="drawing" width="38%">
>    <img src="docs/figures/demo/case1.render.gif" alt="drawing" width="38%">
> </p>

> <p align="center" float="left">
>    <img src="docs/figures/demo/case2.sat.gif" alt="drawing" width="19%">  
>    <img src="docs/figures/demo-density/case2.gif" alt="drawing" width="38%">
>    <img src="docs/figures/demo/case2.render.gif" alt="drawing" width="38%">
> </p>

> <p align="center" float="left">
>    <img src="docs/figures/demo/case3.sat.gif" alt="drawing" width="19%">  
>    <img src="docs/figures/demo-density/case3.gif" alt="drawing" width="38%">
>    <img src="docs/figures/demo/case3.render.gif" alt="drawing" width="38%">
> </p>

> <p align="center" float="left">
>    <img src="docs/figures/demo/case4.sat.gif" alt="drawing" width="19%">  
>    <img src="docs/figures/demo-density/case4.gif" alt="drawing" width="38%">
>    <img src="docs/figures/demo/case4.render.gif" alt="drawing" width="38%">
> </p>

See the [Project Page](https://sat2density.github.io/) for more results and a brief video introduction to Sat2Density.

## Checkpoints Downloading
> Two checkpoints for CVACT and CVUSA can be found from [this url](https://github.com/sat2density/checkpoints/releases). You can also run the following command to download them.
```
bash scripts/download_weights.sh
```

## QuickStart Demo
### Video Synthesis
  #### Example Usage
  ```
  python test.py --yaml=sat2density_cvact \
    --test_ckpt_path=2u87bj8w \
    --task=test_vid \
    --demo_img=demo_img/case1/satview-input.png  \
    --sty_img=demo_img/case1/groundview.image.png  \
    --save_dir=results/case1
  ```
  #### 
We visualize our .vtk shape files with Paraview.

### Illumination Interpolation
<!-- ```
bash inference/quick_demo_interpolation.sh
``` -->
```
python test.py --task=test_interpolation \
--yaml=sat2density_cvact \
--test_ckpt_path=2u87bj8w \
--sty_img1=demo_img/case9/groundview.image.png \
--sty_img2=demo_img/case7/groundview.image.png \
--demo_img=demo_img/case3/satview-input.png \
--save_dir=results/case2
```

## Train & Inference
- *We trained our model using 1 V100 32GB GPU. The training phase will take about 20 hours.*
- *For data preparation, please check out [data.md](dataset/INSTALL.md).*




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
@InProceedings{Qian_2023_ICCV,
    author    = {Qian, Ming and Xiong, Jincheng and Xia, Gui-Song and Xue, Nan},
    title     = {Sat2Density: Faithful Density Learning from Satellite-Ground Image Pairs},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {3683-3692}
}
```

