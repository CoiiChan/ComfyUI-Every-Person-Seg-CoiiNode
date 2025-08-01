import torch
import numpy as np
from PIL import Image
import cv2
import folder_paths
import scipy.ndimage
import os

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("请先安装 ultralytics 包: pip install ultralytics")

def download_url(url, save_path):
    import requests
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

# 自动下载默认模型
ultralytics_segm_path = os.path.join(folder_paths.models_dir, "ultralytics", "segm")
default_model_name = "person_yolov8m-seg.pt"
default_model_path = os.path.join(ultralytics_segm_path, default_model_name)
if not os.path.exists(default_model_path):
    os.makedirs(ultralytics_segm_path, exist_ok=True)
    print(f"[EveryPersonSegSimple] Downloading default model: {default_model_name} Path: {ultralytics_segm_path}")
    download_url(
        "https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8m-seg.pt",
        default_model_path
    )

class EveryPersonSegSimple:
    @classmethod
    def INPUT_TYPES(cls):
        bboxs = ["bbox/"+x for x in folder_paths.get_filename_list("ultralytics_bbox")]
        segms = ["segm/"+x for x in folder_paths.get_filename_list("ultralytics_segm")]
        return {
            "required": {
                "images": ("IMAGE",),
                "yolov_path": (
                    segms,  # 只允许选择 segm/ 下的分割模型
                    {"tooltip": "YOLOv8分割模型文件名"},
                ),
                "confidence": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "drop_area": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 99.0, "step": 1, "tooltip": "最小保留面积百分比（0~99），小于此比例的mask会被丢弃"}),
                "person_fullfil": ("BOOLEAN", {"default": False, "label_on": "person_area", "label_off": "person_masks"}),
                "order": (
                    ["confidence", "large-small", "small-large", "left-right", "right-left", "top-bottom", "bottom-top"],
                    {"default": "confidence", "tooltip": "排序方式"},
                ),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "generate_person_masks"
    CATEGORY = "mask/coiinode"

    def __init__(self):
        self.model = None
        self.model_name = None

    def load_model(self, yolov_path):
        if self.model is None or self.model_name != yolov_path:
            model_file = yolov_path[len("segm/"):] if yolov_path.startswith("segm/") else yolov_path
            model_path = folder_paths.get_full_path("ultralytics_segm", model_file)
            if model_path is None:
                raise ValueError(f"模型文件 '{model_file}' 未找到于 ultralytics/segm 目录下")
            self.model = YOLO(model_path)
            self.model_name = yolov_path

    def generate_person_masks(self, images, yolov_path, confidence, drop_area, person_fullfil, order):
        self.load_model(yolov_path)
        mask_batches = []
        for idx, tensor_image in enumerate(images):
            i = 255.0 * tensor_image.cpu().numpy()
            if i.shape[-1] == 3:
                i = np.dstack((i, np.full((i.shape[0], i.shape[1]), 255)))
            image_pil = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            results = self.model.predict(image_pil, conf=confidence, verbose=False)
            person_masks = []
            img_area = image_pil.width * image_pil.height
            min_area = img_area * (drop_area / 100.0)
            
            # 存储掩码及其属性用于排序
            mask_info = []
            for r in results:
                for j, c in enumerate(r.boxes.cls):
                    if int(c) == 0:  # COCO 0: person
                        if r.masks is not None:
                            mask = r.masks.data[j].cpu().numpy()
                            mask = cv2.resize(mask, (image_pil.width, image_pil.height))
                            mask = (mask > 0.5).astype(np.float32)
                            if mask.sum() >= min_area:
                                # 获取置信度
                                conf = r.boxes.conf[j].cpu().item()
                                
                                # 计算掩码面积
                                area = mask.sum()
                                
                                # 计算边界位置
                                coords = np.where(mask > 0)
                                if len(coords[0]) == 0 or len(coords[1]) == 0:
                                    continue
                                left = np.min(coords[1])
                                right = np.max(coords[1])
                                top = np.min(coords[0])
                                bottom = np.max(coords[0])
                                
                                mask_info.append({
                                    'mask': mask,
                                    'confidence': conf,
                                    'area': area,
                                    'left': left,
                                    'right': right,
                                    'top': top,
                                    'bottom': bottom
                                })
            
            # 根据order参数排序
            if order == 'confidence':
                mask_info.sort(key=lambda x: x['confidence'], reverse=True)
            elif order == 'large-small':
                mask_info.sort(key=lambda x: x['area'], reverse=True)
            elif order == 'small-large':
                mask_info.sort(key=lambda x: x['area'], reverse=False)
            elif order == 'left-right':
                mask_info.sort(key=lambda x: x['left'], reverse=False)
            elif order == 'right-left':
                mask_info.sort(key=lambda x: x['left'], reverse=True)
            elif order == 'top-bottom':
                mask_info.sort(key=lambda x: x['top'], reverse=False)
            elif order == 'bottom-top':
                mask_info.sort(key=lambda x: x['top'], reverse=True)
            
            # 提取排序后的掩码
            person_masks = [info['mask'] for info in mask_info]
            
            if len(person_masks) == 0:
                # 全部被丢弃，输出一张全黑遮罩，shape=[1,1,H,W]
                mask_batch = torch.zeros((1, 1, image_pil.height, image_pil.width), dtype=torch.float32)
            else:
                mask_batch = np.stack(person_masks, axis=0)  # [N, H, W]
                mask_batch = np.expand_dims(mask_batch, axis=1)  # [N, 1, H, W]
                mask_batch = torch.from_numpy(mask_batch).float()
            mask_batches.append(mask_batch)
        # 保证所有 mask_batch shape 一致
        max_h = max([m.shape[2] for m in mask_batches])
        max_w = max([m.shape[3] for m in mask_batches])
        for i in range(len(mask_batches)):
            if mask_batches[i].shape[2] != max_h or mask_batches[i].shape[3] != max_w:
                # resize 到最大尺寸
                mask_batches[i] = torch.nn.functional.interpolate(mask_batches[i], size=(max_h, max_w), mode='nearest')
        if mask_batches:
            all_masks = torch.cat(mask_batches, dim=0)
        else:
            all_masks = torch.zeros((0, 1, 256, 256))

        # 去重叠处理，高置信度优先
        person_bin = (all_masks > 0.5).cpu().numpy().astype(np.uint8)
        N, _, H, W = person_bin.shape
        for i in range(1, N):
            prev_union = np.sum(person_bin[:i, 0], axis=0)
            person_bin[i, 0] = person_bin[i, 0] * (1 - prev_union)
        all_masks = torch.from_numpy(person_bin).to(all_masks.device).type(all_masks.dtype)

        if person_fullfil:
            # --- person_area计算 --- 
            person_bin = (all_masks > 0.5).cpu().numpy().astype(np.uint8)
            N, _, H, W = person_bin.shape
            dist_maps = np.zeros((N, H, W), dtype=np.float32)
            for i in range(N):
                inv_mask = 1 - person_bin[i, 0]
                dist_maps[i] = scipy.ndimage.distance_transform_edt(inv_mask)
            nearest_id = np.argmin(dist_maps, axis=0)  # [H, W]
            person_area = np.zeros((N, 1, H, W), dtype=np.uint8)
            for i in range(N):
                area = ((person_bin[i, 0] == 1) | (nearest_id == i)).astype(np.uint8)
                person_area[i, 0] = area
            person_area = torch.from_numpy(person_area)
            return (person_area,)
        else:
            return (all_masks,)
