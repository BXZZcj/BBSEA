import torch
from torchvision.io import read_image
import os
import cv2
import numpy as np
from plyfile import PlyData, PlyElement


def crop_pcd_single_mask(pointcloud, mask):
    # The last dimension of pointcloud is the size of each pointcloud
    assert pointcloud.shape[-3:-1]==mask.shape[-2:],"the sizes of the pointcloud and the mask should be the same"
    mask=mask.squeeze()
    return pointcloud[mask]


def crop_pcd_with_labels(pointcloud, masks, labels):
    pointcloud_label_map = []
    for mask, label in zip(masks, labels):
        cropped_pointcloud = crop_pcd_single_mask(pointcloud, mask)
        pointcloud_label_map.append({
            'pointcloud': cropped_pointcloud,
            'label': label
        })
    return pointcloud_label_map


def save_color_pc_ply(pointcloud, file_name):
    vertex = np.array([(pointcloud[i, j, 0], pointcloud[i, j, 1], pointcloud[i, j, 2],
                        pointcloud[i, j, 3], pointcloud[i, j, 4], pointcloud[i, j, 5])
                    for i in range(pointcloud.shape[0])
                    for j in range(pointcloud.shape[1])],
                    dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    el = PlyElement.describe(vertex, 'vertex', comments=['vertices with color'])
    PlyData([el], text=True).write(file_name)


class RGBD2PC:
    def __init__(self, camera_factor, fx, fy, cx, cy):
        self.camera_factor=camera_factor
        self.fx=fx
        self.fy=fy
        self.cx=cx
        self.cy=cy


    def transform(self, depth_map):
        """
        Assuming in the depth map, the biggest pixel value represents the nearest distance.
        This assumption will influence the calculation of the Z value.
        """
        height, width = depth_map.shape
        x = torch.arange(width).view(1, -1).repeat(height, 1).float()
        y = torch.arange(height).view(-1, 1).repeat(1, width).float()
        
        x = (x - self.cx) / self.fx
        y = (y - self.cy) / self.fy

        Zc = (255-depth_map.float())/self.camera_factor  

        Xc = x * Zc
        Yc = y * Zc

        pointcloud = torch.stack((Xc, Yc, Zc), dim=-1)
        # pointcloud = pointcloud.cpu().numpy()

        return pointcloud


    def transform_with_color(self, depth_map, rgb_img):
        """
        Assuming in the depth map, the biggest pixel value represents the nearest distance.
        This assumption will influence the calculation of the Z value.
        """
        assert depth_map.shape[-2:]==rgb_img.shape[-2:], "the sizes of depth map and RGB image should be equal"

        pcd_gray=self.transform(depth_map)
        Xc, Yc, Zc = pcd_gray[..., 0], pcd_gray[..., 1], pcd_gray[..., 2]

        R = rgb_img[0].float()/255
        G = rgb_img[1].float()/255
        B = rgb_img[2].float()/255

        pointcloud = torch.stack((Xc, Yc, Zc, R, G, B), dim=-1)
        # pointcloud = pointcloud.cpu().numpy()

        return pointcloud

    

if __name__=="__main__":
    depth_map=read_image("/root/BBSEA/images/lovers_depth.png")[0]
    rgb_img=read_image("/root/BBSEA/images/lovers_rgb.png")

    camera_factor, fx, fy, cx, cy = 4, 700, 700, 170.5, 109.5

    rgbd2pc=RGBD2PC(camera_factor, fx, fy, cx, cy)
    pointcloud=rgbd2pc.transform_with_color(depth_map=depth_map, rgb_img=rgb_img)
    RGBD2PC.save_color_pc_ply(pointcloud=pointcloud,file_name="lovers_point_cloud.ply")