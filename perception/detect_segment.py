import torch
from torchvision.io import read_image
from torchvision.transforms import Resize
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

from utils.visualization import show_box, show_mask, show_points
from config.core import device, SAM_model_path, YOLOv8_model_path


def crop_image(image, mask, visualization=False):
    """
    The element value type of the mask should be the Boolean.
    image: PIL.Image
    mask: numpy.ndarray
    """
    # The 1st dimension of the PIL.Image.size is the width, the 2nd is the height.
    assert image.size[-1:-3:-1] == mask.shape[-2:], "The sizes of the mask and image should be the same."
    
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    ymin, ymax = np.where(rows)[0][0], np.where(rows)[0][-1]
    xmin, xmax = np.where(cols)[0][0], np.where(cols)[0][-1]

    cropped_image = image.crop((xmin, ymin, xmax, ymax))

    if visualization:
        plt.imshow(cropped_image)
        plt.axis('off')
        plt.savefig("cropped_image.png", bbox_inches='tight', pad_inches=0)
        plt.close()

    return cropped_image


class YOLOv8:
    def __init__(self, model):
        self.model=YOLO(model).to(device)
        self.cls_name_map=self.model.names

    def detect(self, image, visualize=False):
        """
        return dict(bbox : name)
        """
        results=self.model(image, verbose=False)

        # Assuming only one image is input, 
        # so the results index is fixed to be zero.
        boxes=results[0].boxes
        xyxys=boxes.xyxy
        cls_name=[self.cls_name_map[int(i.item())] for i in boxes.cls]
        # box_cls_map={tuple(k.cpu().tolist()):j for k,j in zip(xyxys, cls_name)}

        if visualize:
            plt.figure(figsize=(10, 10))
            plt.imshow(image) 
            ax = plt.gca()  
            for i in range(len(xyxys)):
                xyxy = xyxys[i].cpu().numpy() 
                name = cls_name[i]
                show_box(xyxy, ax, name)
            plt.axis('off')
            plt.savefig("detect_segment_yolov8.png")
            plt.close()
        
        return xyxys, cls_name


class SAM:
    def __init__(self, model_type, checkpoint):
        self.model_type=model_type
        self.sam = sam_model_registry[self.model_type](checkpoint)
        self.sam.to(device)
        self.predictor = SamPredictor(self.sam)
        # Ensure the longest side length of the image is 1024
        self.transform = ResizeLongestSide(self.sam.image_encoder.img_size)
    
    def segment_bbox(self, image, bboxes, visualization=False):
        transformed_boxes = self.predictor.transform.apply_boxes_torch(bboxes, image.shape[:2])
        self.predictor.set_image(image)
        masks, _, _ =  self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        if visualization:
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            for mask in masks:
                show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
            for box in bboxes:
                show_box(box.cpu().numpy(), plt.gca())
            plt.axis('off')
            plt.savefig("detect_segment_sam.png")
            plt.close()
        
        return masks


if __name__=="__main__":
    detector=YOLOv8(YOLOv8_model_path["yolov8n"])
    segmentor=SAM("vit_h", SAM_model_path["vit_h"])

    image_path="/root/BBSEA/images/lovers_rgb.png"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    xyxys, cls_name = detector.detect(image, visualize=True)
    print("the detected bboxes are:")
    box_cls_map={tuple(k.cpu().tolist()):j for k,j in zip(xyxys, cls_name)}
    for i in box_cls_map:
        print(f"{i}:{box_cls_map[i]}")

    masks=segmentor.segment_bbox(image, xyxys, visualization=True)
    masks, cls_name