import torch
import clip
import PIL
from PIL import Image
from torchvision.io import read_image

from config.core import device


class CLIP_classify:
    def __init__(self, backbone="ViT-B/32"):
        self.model, self.preprocess = clip.load(backbone, device=device)

    def classify(self, cls_names, image: PIL.Image):
        cls_names_tokenized = clip.tokenize(cls_names).to(device)
        image = self.preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits_per_image, logits_per_text = self.model(image, cls_names_tokenized)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        return cls_names[probs.argmax()]


if __name__=="__main__":
    from detect_segment import YOLOv8, SAM, crop_image

    detector=YOLOv8(YOLOv8_model_path["yolov8n"])
    segmentor=SAM("vit_h", SAM_model_path["vit_h"])

    image_path="/root/BBSEA/images/lovers_rgb.png"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    xyxys, cls_names = detector.detect(image, visualize=True)

    masks=segmentor.segment_bbox(image, xyxys, visualization=True)

    cropped_image=crop_image(Image.fromarray(image), masks[4].squeeze().cpu().numpy(), visualization=True)

    cls_names=["drawer", "bin", "block", "stick", "man", "woman"]
    clip_classify=CLIP_classify()
    pred_cls=clip_classify.classify(cls_names, cropped_image)
    print(pred_cls)