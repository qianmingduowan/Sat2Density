import os  
import cv2  
from PIL import Image  

def image_to_video(img_dir,image_names, media_path):
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    fps = 20  
    image = Image.open(os.path.join( img_dir , image_names[0]))
    media_writer = cv2.VideoWriter(media_path, fourcc, fps, image.size)
    for image_name in image_names:
        im = cv2.imread(os.path.join(img_dir, image_name))
        media_writer.write(im)
        print(image_name, 'combined')
    media_writer.release()
    print('end')

def img_pair2vid(sat_list,grd_list,angle_list=None,media_path= 'output.mp4'):
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    out = cv2.VideoWriter(media_path, fourcc, 12.0, (512, 256))
    out_sat = cv2.VideoWriter(media_path.replace('.mp4','_sat.mp4'), fourcc, 12.0, (389, 389))
    assert len(sat_list) == len(grd_list)
    for i  in range(len(sat_list)):

        img1 = cv2.imread(os.path.join( img_dir , sat_list[i]))
        img2 = cv2.imread(os.path.join( img_dir , grd_list[i]))
        img3 = cv2.imread(os.path.join( img_dir , grd_list[i].replace('.png','_depth.png')))


        if angle_list!=None:
            angle = angle_list[i]
            left_pixel = int((angle/180)*256)
            if angle<0:
                img2 = cv2.hconcat([img2[:,left_pixel:,:],img2[:,:left_pixel,:]])
                img3= cv2.hconcat([img3[:,left_pixel:,:],img3[:,:left_pixel,:]])
            else:
                img2 = cv2.hconcat([img2[:,left_pixel:,:],img2[:,:left_pixel,:]])
                img3 = cv2.hconcat([img3[:,left_pixel:,:],img3[:,:left_pixel,:]])
        merged_image = cv2.vconcat([img2,img3])
        out.write(merged_image)
        out_sat.write(img1)
    out.release()
    out_sat.release()

if __name__=='__main__':
    import csv
    img_dir = 'vis_video'
    img_list = sorted(os.listdir(img_dir))
    sat_list = []
    grd_list = []
    for img in img_list:
        if '.png' in img:
            if 'satdepth'  in img:
                continue
            if 'grdView_pano.png' in img:
                continue
            if 'grdView' in img:
                if '_depth.png' not in img:
                    grd_list.append(img)
            elif 'satView' in img:
                sat_list.append(img)
    sat_list = sat_list[:-1]
    grd_list = grd_list[:-1]
    media_path = os.path.join(img_dir,'output_cat.mp4')
    angle_list = []
    with open(os.path.join(img_dir,'pixels.csv') , 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            angle = float(row['angle'])
            angle_list.append(angle)
    print(angle_list)

    img_pair2vid(sat_list,grd_list,angle_list,media_path= media_path)
    print('save 2 ',media_path)