import cv2
import numpy as np
import torch
import torchvision

class Equirectangular():
    """
    Random sample a panorama image into a perspective view
    take https://github.com/fuenwang/Equirec2Perspec/blob/master/Equirec2Perspec.py as a reference
    """
    def __init__(self, width = 256, height = 256, FovX = 100, theta = [0, 0]):
        """
        width: output image's width
        height: output image's height
        FovX: perspective camera FOV on x-axis (degree)
        theta: theta field where img's theta degree from 
        """
        self.theta = theta
        self.width = width
        self.height = height
        self.type = type

        #create x-axis coordinates and corresponding y-axis coordinates
        x = np.arange(width)
        y = np.arange(height)
        x, y = np.meshgrid(x, y) 
        
        #create homogenerous coordinates
        z = np.ones_like(x)
        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
        
        #translation matrix
        f = 0.5 * width * 1 / np.tan(np.radians(FovX/2))
        # cx = (width - 1) / 2.0
        # cy = (height - 1) / 2.0
        cx = (width) / 2.0
        cy = (height) / 2.0        
        K = np.array([
                [f, 0, cx],
                [0, f, cy],
                [0, 0,  1],
            ], np.float32)
        K_inv = np.linalg.inv(K)
        xyz = xyz @ K_inv.T
        self.xyz = xyz  ### self.xyz is the direction of the each ray in the camera space when camera is fixed



    def __call__(self, img1): 
        batch = img1.shape[0]
        PHI, THETA = self.getRandomRotation(batch)
        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        x_axis = np.array([1.0, 0.0, 0.0], np.float32)
        #rotation matrix
        xy_grid = []
        for i in range(batch):
            R1, _ = cv2.Rodrigues(y_axis * np.radians(PHI[i]))
            R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(THETA[i]))
            R = R2 @ R1
            #rotate
            xyz = self.xyz @ R.T  ### ### xyz is the direction of the each ray in the camera space when camera is rotate
            norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
            xyz_norm = xyz / norm
            
            #transfer to image coordinates
            xy = self.xyz2xy(xyz_norm)
            device = img1.device
            xy = torch.from_numpy(xy).to(device).unsqueeze(0)
            xy_grid.append(xy)
        xy = torch.cat(xy_grid,dim=0)

        #resample
        return xy

    def xyz2xy(self, xyz_norm):
        #normlize
        x = xyz_norm[..., 0]
        y = xyz_norm[..., 1]
        z = xyz_norm[..., 2]

        lon = np.arctan2(x, z)
        lat = np.arcsin(y)
        ### transfer to the lon and lat

        X = lon / (np.pi)
        Y = lat / (np.pi) * 2
        xy = np.stack([X, Y], axis=-1)
        xy = xy.astype(np.float32)
        
        return xy

    def getRandomRotation(self,batch_size):
        # phi = np.random.rand(batch_size) * 360 -180
        phi = np.random.randint(-180,180,batch_size)
        assert(self.theta[0]<self.theta[1])
        theta = np.random.randint(self.theta[0],self.theta[1],batch_size)
        # theta = np.random.rand(batch_size)*(self.theta[1]-self.theta[0])-self.theta[0]
        return phi, theta


if __name__=='__main__':
    # test demo
    e = Equirectangular(theta=[0., 40.],width = 64, height = 64,FovX = 100)
    img = cv2.imread('dataset/CVACT/streetview/__-DFIFxvZBCn1873qkqXA_grdView.png')[:,:,::-1]/255.0
    img = img.transpose(2, 0, 1).astype(np.float32)
    
    img = torch.from_numpy(img).unsqueeze(0).repeat(10, 1, 1, 1)
    equ= e(img) 
    # print(PHI, THETA)   
    torchvision.utils.save_image(torch.nn.functional.grid_sample(img, equ.float(), align_corners = True)*0.99, 'test_30.png')