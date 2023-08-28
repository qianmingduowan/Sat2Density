import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.image import imread
import csv,os,sys

data = sys.argv[1]
assert data.endswith('satView_polish.png')
img_path = os.path.join('dataset/CVACT/satview_correct',data)

# img_path = './dataset/CVACT/satview_correct/__-DFIFxvZBCn1873qkqXA_satView_polish.png'
csv_path = 'vis_video/pixels.csv'
select_points = [28, 44, 53]

x_list,y_list = [],[]
x_whole,y_whole = [],[]
with open(csv_path, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for i,row in enumerate(reader):
        x,y = float(row['x']),float(row['y']) 
        if  i in select_points:
            x_list.append(x)
            y_list.append(y)
            print(i,x,y)
        x_whole.append(x)
        y_whole.append(y)
fig, ax = plt.subplots()


img = imread(img_path)
plt.imshow(img)
plt.plot(x_whole, y_whole, 'r-',label='Smooth curve', linewidth=4)
plt.scatter(x_list,y_list,marker='o', s=0, color='red')
plt.axis('off')
plt.xlim([0, 256])
plt.ylim([256, 0])
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig('point_curve.png', bbox_inches='tight', pad_inches=0)
