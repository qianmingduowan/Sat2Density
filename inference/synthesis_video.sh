### a demo for synthesis ground video
name = __-DFIFxvZBCn1873qkqXA_satView_polish.png

# select point
# First find a starting point, then press and hold the left mouse button, draw any shape, 
# then release the left mouse button, and press 'q' on the keyboard to end the point selection process

#  better select regions near the center of the satellite image. 'q' to end select point.
python inference/select_points.py ${name}

# inference
# if you want use illumination from another image , you could add --sty_img=WsKPDHEgLwrhrJXcUU34xA_grdView.png
CUDA_VISIBLE_DEVICES=0 python offline_train_test.py --task=test_vid --yaml=sat2density_cvact --test_ckpt_path=2u87bj8w --demo_img=${name}

# make video
python img2vid.py