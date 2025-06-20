# ComfyUI-Every-Person-Seg-CoiiNode
[English](https://github.com/CoiiChan/ComfyUI-Every-Person-Seg-CoiiNode/blob/main/README.md)|[中文](https://github.com/CoiiChan/ComfyUI-Every-Person-Seg-CoiiNode/blob/main/README_CN.md)

## Project Introduction
A tool that provides the ability to split fine-grained contours of people one by one for multi-person usage scenarios.

## Development Reason
There is a demand for multi-person generation in ComfyUI applications, but many tools for segmenting person IDs use bbox, which has the problem of not being able to separate multiple people cleanly. Therefore, this custom node plugin was created. It provides necessary implementation capabilities for scenarios such as multi-person loop inpaint and face swapping.

## ComfyUI Node Description

EveryPersonSegSimple Node

![showit](https://github.com/CoiiChan/ComfyUI-Every-Person-Seg-CoiiNode/blob/main/example/exampler_everypersonsimple.gif)

EveryPersonSegSimple Node with person_fullfil:True

![showit](https://github.com/CoiiChan/ComfyUI-Every-Person-Seg-CoiiNode/blob/main/example/exampler_person_area.gif)

EveryPersonSegDetail Node

![showit](https://github.com/CoiiChan/ComfyUI-Every-Person-Seg-CoiiNode/blob/main/example/exampler_everypersonsegdetail.gif)


### EveryPersonSegSimple


#### Input Parameters
- **images**: Input image
- **yolov_path**: YOLOv8 segmentation model path (default uses `person_yolov8m-seg.pt`)
- **confidence**: Detection confidence threshold (0.0 ~ 1.0)
- **drop_area**: Minimum reserved area percentage (0 ~ 99)
- **person_fullfil**: Whether to generate complete portrait neighboring fill area (person_area) or original portrait individual mask (person_masks)

#### Output
- **mask**: Portrait mask collection

## Usage Example

Drag the workflow example/workflowexample_everypersonseg.json into ComfyUI to use:

EveryPersonSegSimple
![Workflow Example](https://github.com/CoiiChan/ComfyUI-Every-Person-Seg-CoiiNode/blob/main/example/exampler_everypersonsimple.png)

EveryPersonSegDetail
![Workflow Example](https://github.com/CoiiChan/ComfyUI-Every-Person-Seg-CoiiNode/blob/main/example/exampler_everypersonsegdetail.png)

## Dependencies
- mediapipe>=0.10.13
- ultralytics>=8.0.0
- numpy>=1.21.0
- opencv-python>=4.5.0
- Pillow>=9.0.0
- torch>=1.10.0
- scipy>=1.7.0


## Manual Model Download

If automatic download fails, please manually download the following models and place them in the specified paths:

### YOLOv8 Segmentation Model
- **Download URL**: https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8m-seg.pt
- **Save Path**: `ComfyUI/models/ultralytics/segm/person_yolov8m-seg.pt`

### MediaPipe Portrait Segmentation Model
- **Download URL**: https://huggingface.co/yolain/selfie_multiclass_256x256/resolve/main/selfie_multiclass_256x256.tflite?download=true
- **Save Path**: `ComfyUI/models/mediapipe/selfie_multiclass_256x256.tflite`

## Acknowledgments
### EveryPersonSegDetail

This node combines YOLOv8 segmentation model and MediaPipe portrait segmentation model to provide more detailed portrait masks (supporting fine segmentation of face, hair, body, clothes and other parts). The mask contour optimization part of the code refers to the [djbielejeski/a-person-mask-generator](https://github.com/djbielejeski/a-person-mask-generator) project.

#### Input Parameters
- **images**: Input image
- **face_mask/hair_mask/body_mask/clothes_mask/background_mask**: Select mask types to generate
- **refine_mask**: Whether to enable mask contour optimization
- **confidence**: Detection confidence threshold
- **yolov_path**: YOLOv8 segmentation model path
- **drop_area**: Minimum reserved area percentage

#### Output
- **person_masks**: Refined portrait part mask collection
  
---
[![CoiiChan](https://avatars.githubusercontent.com/u/49615294?v=4)](https://github.com/CoiiChan)

## Acknowledgments
The mask contour optimization part of the EveryPersonSegDetail node in this project is modified and implemented based on the [djbielejeski/a-person-mask-generator](https://github.com/djbielejeski/a-person-mask-generator) project.
