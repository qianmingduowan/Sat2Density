import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from matplotlib.image import imread
import numpy as np
import csv,os
from scipy.interpolate import interp1d
import sys
data = sys.argv[1]
assert data.endswith('satView_polish.png')
dirs = os.path.join('dataset/CVACT/satview_correct',data)
if not os.path.exists(dirs):
    dirs = dirs.replace('dataset/CVACT','demo_img')
sav_pth = 'vis_video'
if not os.path.exists(sav_pth):
    os.mkdir(sav_pth)

img = imread(dirs)

fig = plt.figure()
fig.set_size_inches(1,1,forward=False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
ax.imshow(img)

coords = []

def ondrag(event):
    if event.button != 1:
        return
    x, y = int(event.xdata), int(event.ydata)
    coords.append((x, y))
    ax.plot([x], [y], 'o', color='red')
    fig.canvas.draw_idle()
fig.add_axes(ax)
cursor = Cursor(ax, useblit=True, color='red', linewidth=1)
fig.canvas.mpl_connect('motion_notify_event', ondrag)
plt.show()
plt.close()


unique_lst = list(dict.fromkeys(coords))
pixels = []
for x in coords:
    if x in unique_lst:
        if x not in pixels:
            pixels.append(x)
print(pixels)

###########################################

from scipy.interpolate import splprep, splev

points = pixels
points = np.array(points)
tck, u = splprep(points.T, s=25, per=0)
u_new = np.linspace(u.min(), u.max(), 80)
x_new, y_new = splev(u_new, tck)

plt.plot(points[:,0], points[:,1], 'ro', label='Original curve')
plt.plot(x_new, y_new, 'b-', label='Smooth curve')
plt.legend()
plt.show()
plt.close()

fig, ax = plt.subplots()


pixels  = [tuple(sublist[:2]) for sublist in zip(x_new,y_new)]
###########################################
img = imread(dirs)
fig, ax = plt.subplots()
ax.set_xticks([])
ax.set_yticks([])
ax.imshow(img)
plt.plot(x_new, y_new, 'r-', label='Smooth curve')
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(os.path.join(sav_pth,os.path.basename(dirs)).replace('.png','_sat_track.png'),bbox_inches='tight', pad_inches=0)
plt.close()

###########################################
angle_list = []
for i,pixel in enumerate(pixels[:-1]):
    img = imread(dirs)

    x1, y1 = pixel
    x2, y2 = pixels[i+1]
    dx, dy = x2 - x1, y2 - y1
    angle_save = np.degrees(np.arctan2(dy, dx))+90
    if angle_save>180:
        angle_save = angle_save-360
    angle_list.append(angle_save)
    length = np.sqrt(dx ** 2 + dy ** 2)
    angle = np.arctan2(dy, dx) * 180 / np.pi
    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(img)
    ax.arrow(x1, y1, dx*10, dy*10, color='red', width=length, head_width=4*length, head_length=5*length)
    
    name = '_sat'+'%05d' % int(i) + ".png"
    plt.savefig(os.path.join(sav_pth,os.path.basename(dirs)).replace('.png',name),bbox_inches='tight')
    plt.close()


with open( os.path.join(sav_pth,'pixels.csv'), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['x', 'y','angle'])
    for i, (x, y) in enumerate(pixels[:-1]):
        writer.writerow([x, y,angle_list[i]])
print('save to pixels.csv',len(pixels[:-1]))