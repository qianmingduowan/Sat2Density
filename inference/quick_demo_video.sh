sat_name=__-DFIFxvZBCn1873qkqXA_satView_polish.png

sty_name=VAMM6sIEbYAY5E6ZD_RMKg_grdView.png

# select point
# First find a starting point, then press and hold the left mouse button, draw any shape, 
# then release the left mouse button, and press 'q' on the keyboard to end the point selection process

#  better select regions near the center of the satellite image. 'q' to end select point.
# python inference/select_points.py ${sat_name}

# inference
# if you want use illumination from another image , you could add --sty_img=WsKPDHEgLwrhrJXcUU34xA_grdView.png
CUDA_VISIBLE_DEVICES=0 python offline_train_test.py --yaml=sat2density_cvact \
--test_ckpt_path=2u87bj8w \
--task=test_vid \
--demo_img=${sat_name} --sty_img=${sty_name} \
--data.root=demo_img


# make video
python inference/img2vid.py

#  visualize  vis_video/volume_data.vtk with ParaView


# python test.py --yaml=sat2density_cvact \
#     --test_ckpt_path=2u87bj8w \
#     --task=test_vid \
#     --demo_img=__-DFIFxvZBCn1873qkqXA_satView_polish.png \
#     --sty_img=VAMM6sIEbYAY5E6ZD_RMKg_grdView.png \
#     --data.root=demo_img
