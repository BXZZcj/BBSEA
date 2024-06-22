import torch

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

SAM_model_path = {
    "default": "path/to/vit_h",
    "vit_h": "/root/autodl-tmp/checkpoints/SAM/sam_vit_h_4b8939.pth",
    "vit_l": "path/to/vit_l",
    "vit_b": "path/to/vit_b",
}

YOLOv8_model_path = {
    "yolov8n":"/root/autodl-tmp/checkpoints/yolov8/yolov8n.pt",
    "yolov8x":"/root/autodl-tmp/checkpoints/yolov8/yolov8x.pt",
}

prompts_path = {
    "task_propose":"/root/BBSEA/config/prompts/task_propose.txt",
    "task_decompose":"/root/BBSEA/config/prompts/task_decompose.txt",
    "success_infer":"/root/BBSEA/config/prompts/success_infer.txt",
}

gpt_api_key="sk-oxvoz5HymVZw5buJ9GT5wNsW2FWN4Kbyva2ENI4wbE47TWUv"