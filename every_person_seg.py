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

class GenderDetector:
    def __init__(self, models_dir):
        self.models_dir = models_dir
        self.gender_net = None
        self.face_cascade = None
        self.mean_values = (78.4263377603, 87.7689143744, 114.895847746)
    
    def load_models(self):
        if self.gender_net is not None:
            return
        
        gender_proto = os.path.join(self.models_dir, "gender_deploy.prototxt")
        gender_model = os.path.join(self.models_dir, "gender_net.caffemodel")
        
        if not os.path.exists(gender_model):
            print(f"[GenderDetector] Downloading gender detection models...")
            os.makedirs(self.models_dir, exist_ok=True)
            download_url(
                "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/gender_deploy.prototxt",
                gender_proto
            )
            download_url(
                "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/gender_net.caffemodel",
                gender_model
            )
        
        self.gender_net = cv2.dnn.readNet(gender_model, gender_proto)
        
        cascade_file = os.path.join(self.models_dir, "haarcascade_frontalface_default.xml")
        if not os.path.exists(cascade_file):
            cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            print(f"[GenderDetector] Downloading Haar cascade model...")
            download_url(cascade_url, cascade_file)
        
        self.face_cascade = cv2.CascadeClassifier(cascade_file)
    
    def detect(self, image_pil, box_coords):
        """返回性别: 0=女, 1=男, None=未检测到"""
        gender, _ = self.detect_with_prob(image_pil, box_coords)
        return gender
    
    def detect_with_prob(self, image_pil, box_coords):
        """返回 (None, 性别得分): (None, 0.0-1.0)"""
        x1, y1, x2, y2 = map(int, box_coords)
        if x2 <= x1 or y2 <= y1:
            return None, None
        
        person_crop = np.array(image_pil.crop((x1, y1, x2, y2)))
        if person_crop.size == 0:
            return None, None
        
        # 确保转换为RGB格式（去除alpha通道）
        if person_crop.shape[2] == 4:
            person_crop = person_crop[:, :, :3]
        
        gray = cv2.cvtColor(person_crop, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        if len(faces) == 0:
            return None, None
        
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        fx, fy, fw, fh = faces[0]
        
        face_img = person_crop[fy:fy+fh, fx:fx+fw]
        if face_img.size == 0:
            return None, None
        
        try:
            face_img = cv2.resize(face_img, (227, 227))
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), self.mean_values, swapRB=False)
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            # 返回性别得分
            gender_score = gender_preds[0][1]  # 假设第二个值是我们需要的得分
            return None, gender_score
        except Exception as e:
            return None, None

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
                    ["confidence", "large-small", "small-large", "left-right", "right-left", "top-bottom", "bottom-top", "female-male", "male-female"],
                    {"default": "confidence", "tooltip": "排序方式"},
                ),
                "combine": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "generate_person_masks"
    CATEGORY = "mask/coiinode"

    def __init__(self):
        self.model = None
        self.model_name = None
        opencv_dnn_path = os.path.join(folder_paths.models_dir, "opencv_dnn")
        self.gender_detector = GenderDetector(opencv_dnn_path)
        self.gender_detector.load_models()

    def load_model(self, yolov_path):
        if self.model is None or self.model_name != yolov_path:
            model_file = yolov_path[len("segm/"):] if yolov_path.startswith("segm/") else yolov_path
            model_path = folder_paths.get_full_path("ultralytics_segm", model_file)
            if model_path is None:
                raise ValueError(f"模型文件 '{model_file}' 未找到于 ultralytics/segm 目录下")
            self.model = YOLO(model_path)
            self.model_name = yolov_path

    def generate_person_masks(self, images, yolov_path, confidence, drop_area, person_fullfil, order, combine):
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
                                
                                # 获取边界框坐标
                                box = r.boxes.xyxy[j].cpu().numpy()
                                x1, y1, x2, y2 = box
                                
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
                                
                                # 性别检测
                                gender_score = None
                                if order in ['female-male', 'male-female']:
                                    _, gender_score = self.gender_detector.detect_with_prob(image_pil, box)
                                
                                mask_info.append({
                                    'mask': mask,
                                    'confidence': conf,
                                    'area': area,
                                    'left': left,
                                    'right': right,
                                    'top': top,
                                    'bottom': bottom,
                                    'gender_score': gender_score
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
            elif order == 'female-male':
                # 按性别得分从高到低排序，未检测到的放最后
                mask_info.sort(key=lambda x: (x['gender_score'] is None, -x['gender_score'] if x['gender_score'] is not None else 0))
            elif order == 'male-female':
                # 按性别得分从低到高排序，未检测到的放最后
                mask_info.sort(key=lambda x: (x['gender_score'] is None, x['gender_score'] if x['gender_score'] is not None else 1))
            
            # 打印排序结果
            if order in ['female-male', 'male-female']:
                print(f"[GenderDetector] 排序方式: {order}")
                for i, info in enumerate(mask_info):
                    score_str = f"(得分: {info['gender_score']:.2f})" if info['gender_score'] is not None else "(未检测到)"
                    print(f"[GenderDetector] 第{i+1}个: {score_str}")
            
            # 提取排序后的掩码
            person_masks = [info['mask'] for info in mask_info]
            
            if combine:
                # 将所有掩码相加（取并集）
                if len(person_masks) == 0:
                    mask_batch = torch.zeros((1, 1, image_pil.height, image_pil.width), dtype=torch.float32)
                else:
                    combined_mask = np.zeros_like(person_masks[0])
                    for mask in person_masks:
                        combined_mask = np.logical_or(combined_mask, mask).astype(np.float32)
                    mask_batch = torch.from_numpy(combined_mask).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            else:
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

        if combine:
            # 对于combine模式，直接返回组合后的掩码
            return (all_masks,)

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
