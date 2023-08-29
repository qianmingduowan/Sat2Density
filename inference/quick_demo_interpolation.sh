CUDA_VISIBLE_DEVICES=0 python offline_train_test.py --task=test_interpolation \
--yaml=sat2density_cvact \
--test_ckpt_path=2u87bj8w \
--sty_img1=YL81FiK9PucIvAkr1FHkpA_grdView.png \
--sty_img2=pdZmLHYEhe2PHj_8-WHMhw_grdView.png \
--demo_img=VAMM6sIEbYAY5E6ZD_RMKg_satView_polish.png \
--data.root=demo_img


python inference/img2vid_interpolation.py