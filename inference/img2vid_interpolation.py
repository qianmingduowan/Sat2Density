import os  
import cv2  
from PIL import Image  


def img_pair2vid(sat_list,media_path= 'interpolation.mp4'):
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    out = cv2.VideoWriter(media_path, fourcc, 12.0, (512, 128))
    for i  in range(len(sat_list)):

        img1 = cv2.imread(os.path.join( img_dir , sat_list[i]))

        out.write(img1)
    out.release()

if __name__=='__main__':
    import csv
    img_dir = 'vis_interpolation'
    img_list = sorted(os.listdir(img_dir))
    sat_list = []
    for img in img_list:
        sat_list.append(img)
    media_path = os.path.join(img_dir,'interpolation.mp4')

    img_pair2vid(sat_list,media_path= media_path)
    print('save 2 ',media_path)