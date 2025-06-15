import math
import os
import sys
import torch
import numpy as np
from PIL import Image
import cv2
import folder_paths
import scipy.ndimage

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("请先安装 ultralytics 包: pip install ultralytics")
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def get_a_person_mask_generator_model_path() -> str:
    model_folder_name = "mediapipe"
    model_name = "selfie_multiclass_256x256.tflite"
    model_folder_path = os.path.join(folder_paths.models_dir, model_folder_name)
    model_file_path = os.path.join(model_folder_path, model_name)
    if not os.path.exists(model_file_path):
        def download_url(url, save_path):
            import requests
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(save_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        model_url = "https://huggingface.co/yolain/selfie_multiclass_256x256/resolve/main/selfie_multiclass_256x256.tflite?download=true"
        print(f"Downloading '{model_name}' model")
        os.makedirs(model_folder_path, exist_ok=True)
        download_url(model_url, model_file_path)
    return model_file_path

class EveryPersonSegDetail:
    @classmethod
    def INPUT_TYPES(cls):
        false_widget = (
            "BOOLEAN",
            {"default": False, "label_on": "enabled", "label_off": "disabled"},
        )
        true_widget = (
            "BOOLEAN",
            {"default": True, "label_on": "enabled", "label_off": "disabled"},
        )
        segms = ["segm/"+x for x in folder_paths.get_filename_list("ultralytics_segm")]
        return {
            "required": {
                "images": ("IMAGE",),
                "face_mask": true_widget,
                "background_mask": false_widget,
                "hair_mask": false_widget,
                "body_mask": false_widget,
                "clothes_mask": false_widget,
                "refine_mask": true_widget,
                "confidence": ("FLOAT", {"default": 0.4, "min": 0.01, "max": 1.0, "step": 0.01}),
                "yolov_path": (
                    segms,
                    {"tooltip": "YOLOv8分割模型文件名"},
                ),
                "drop_area": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 99.0, "step": 1, "tooltip": "最小保留面积百分比（0~99），小于此比例的mask会被丢弃"}),
            },
        }

    CATEGORY = "mask/coiinode"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("person_masks",)
    FUNCTION = "generate_person_masks"

    def __init__(self):
        get_a_person_mask_generator_model_path()
        self.mediapipe_model = None
        self.yolo_model = None
        self.yolo_model_name = None

    def get_mediapipe_mask(self, images, face_mask, background_mask, hair_mask, body_mask, clothes_mask, confidence, refine_mask):
        a_person_mask_generator_model_path = get_a_person_mask_generator_model_path()
        a_person_mask_generator_model_buffer = None
        with open(a_person_mask_generator_model_path, "rb") as f:
            a_person_mask_generator_model_buffer = f.read()
        image_segmenter_base_options = BaseOptions(
            model_asset_buffer=a_person_mask_generator_model_buffer
        )
        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=image_segmenter_base_options,
            running_mode=VisionRunningMode.IMAGE,
            output_category_mask=True,
        )
        mask_images = []
        with ImageSegmenter.create_from_options(options) as segmenter:
            for tensor_image in images:
                i = 255.0 * tensor_image.cpu().numpy()
                if i.shape[-1] == 3:
                    i = np.dstack((i, np.full((i.shape[0], i.shape[1]), 255)))
                image_pil = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                media_pipe_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=np.asarray(image_pil))
                segmented_masks = segmenter.segment(media_pipe_image)
                masks = []
                if background_mask:
                    masks.append(segmented_masks.confidence_masks[0].numpy_view() > confidence)
                if hair_mask:
                    masks.append(segmented_masks.confidence_masks[1].numpy_view() > confidence)
                if body_mask:
                    masks.append(segmented_masks.confidence_masks[2].numpy_view() > confidence)
                if face_mask:
                    masks.append(segmented_masks.confidence_masks[3].numpy_view() > confidence)
                if clothes_mask:
                    masks.append(segmented_masks.confidence_masks[4].numpy_view() > confidence)
                if masks:
                    mask = np.zeros_like(masks[0], dtype=np.float32)
                    for m in masks:
                        mask = np.logical_or(mask, m)
                else:
                    mask = np.zeros((image_pil.height, image_pil.width), dtype=np.float32)
                mask_images.append(torch.from_numpy(mask).float()[None, ...])
        return torch.cat(mask_images, dim=0)  # [B, H, W]

    def get_person_area(self, images, yolov_path, confidence, drop_area):
        # 只支持 segm/ 前缀
        model_file = yolov_path[len("segm/"):] if yolov_path.startswith("segm/") else yolov_path
        model_path = folder_paths.get_full_path("ultralytics_segm", model_file)
        if model_path is None:
            raise ValueError(f"模型文件 '{model_file}' 未找到于 ultralytics/segm 目录下")
        if self.yolo_model is None or self.yolo_model_name != yolov_path:
            self.yolo_model = YOLO(model_path)
            self.yolo_model_name = yolov_path
        mask_batches = []
        for idx, tensor_image in enumerate(images):
            i = 255.0 * tensor_image.cpu().numpy()
            if i.shape[-1] == 3:
                i = np.dstack((i, np.full((i.shape[0], i.shape[1]), 255)))
            image_pil = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            results = self.yolo_model.predict(image_pil, conf=confidence, verbose=False)
            person_masks = []
            img_area = image_pil.width * image_pil.height
            min_area = img_area * (drop_area / 100.0)
            for r in results:
                for j, c in enumerate(r.boxes.cls):
                    if int(c) == 0:
                        if r.masks is not None:
                            mask = r.masks.data[j].cpu().numpy()
                            mask = cv2.resize(mask, (image_pil.width, image_pil.height))
                            mask = (mask > 0.5).astype(np.float32)
                            if mask.sum() >= min_area:
                                person_masks.append(mask)
            if len(person_masks) == 0:
                mask_batch = torch.zeros((1, 1, image_pil.height, image_pil.width), dtype=torch.float32)
            else:
                mask_batch = np.stack(person_masks, axis=0)
                mask_batch = np.expand_dims(mask_batch, axis=1)
                mask_batch = torch.from_numpy(mask_batch).float()
            mask_batches.append(mask_batch)
        max_h = max([m.shape[2] for m in mask_batches])
        max_w = max([m.shape[3] for m in mask_batches])
        for i in range(len(mask_batches)):
            if mask_batches[i].shape[2] != max_h or mask_batches[i].shape[3] != max_w:
                mask_batches[i] = torch.nn.functional.interpolate(mask_batches[i], size=(max_h, max_w), mode='nearest')
        if mask_batches:
            all_masks = torch.cat(mask_batches, dim=0)
        else:
            all_masks = torch.zeros((0, 1, 256, 256))
        # 去重叠
        person_bin = (all_masks > 0.5).cpu().numpy().astype(np.uint8)
        N, _, H, W = person_bin.shape
        for i in range(1, N):
            prev_union = np.sum(person_bin[:i, 0], axis=0)
            person_bin[i, 0] = person_bin[i, 0] * (1 - prev_union)
        all_masks = torch.from_numpy(person_bin).to(all_masks.device).type(all_masks.dtype)
        # person_area
        person_bin = (all_masks > 0.5).cpu().numpy().astype(np.uint8)
        N, _, H, W = person_bin.shape
        dist_maps = np.zeros((N, H, W), dtype=np.float32)
        for i in range(N):
            inv_mask = 1 - person_bin[i, 0]
            dist_maps[i] = scipy.ndimage.distance_transform_edt(inv_mask)
        nearest_id = np.argmin(dist_maps, axis=0)
        person_area = np.zeros((N, 1, H, W), dtype=np.uint8)
        for i in range(N):
            area = ((person_bin[i, 0] == 1) | (nearest_id == i)).astype(np.uint8)
            person_area[i, 0] = area
        return torch.from_numpy(person_area)  # [N, 1, H, W]

    def generate_person_masks(self,
            images,
            face_mask,
            background_mask,
            hair_mask,
            body_mask,
            clothes_mask,
            refine_mask,
            confidence,
            yolov_path,
            drop_area,
        ):
        # mediapipe mask
        mp_mask = self.get_mediapipe_mask(
            images=images,
            face_mask=face_mask,
            background_mask=background_mask,
            hair_mask=hair_mask,
            body_mask=body_mask,
            clothes_mask=clothes_mask,
            confidence=confidence,
            refine_mask=refine_mask,
        )  # [B, H, W]
        # yolo person_area
        person_area = self.get_person_area(images, yolov_path, confidence, drop_area)  # [N, 1, H, W]
        # 取交集
        out_masks = []
        for i in range(person_area.shape[0]):
            # 取交集，保证输出shape为[N, 1, H, W]
            mask = person_area[i, 0] * mp_mask[0]  # 默认只取batch 0
            out_masks.append(mask[None, None, ...])
        if out_masks:
            out_masks = torch.cat(out_masks, dim=0)
        else:
            out_masks = torch.zeros((0, 1, mp_mask.shape[1], mp_mask.shape[2]), dtype=torch.float32)
        return (out_masks,)
