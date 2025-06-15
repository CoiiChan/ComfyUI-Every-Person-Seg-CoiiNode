# comfyui-every-person-seg-coii
## 项目介绍
为多人使用场景提供人物精细轮廓逐一拆分能力的工具。

## 开发原因
多人化的生成在comfyui的应用中存在需求，但是有很多分割人物id的工具使用bbox，bbox存在多人不能拆除干净，因此制作该custom node插件。对于在多人循环inpaint，换脸等场景提供必要实现能力

## ComfyUI 节点说明

EveryPersonSegSimple Node

![showit](https://github.com/CoiiChan/comfyui-every-person-seg-coii/blob/main/example/exampler_person_area.gif)

EveryPersonSegSimple Node person_fullfil:True

![showit](https://github.com/CoiiChan/comfyui-every-person-seg-coii/blob/main/example/exampler_everypersonsimple.gif)

EveryPersonSegDetail Node

![showit](https://github.com/CoiiChan/comfyui-every-person-seg-coii/blob/main/example/exampler_everypersonsegdetail.gif)


### EveryPersonSegSimple


#### 输入参数
- **images**: 输入图像
- **yolov_path**: YOLOv8 分割模型路径（默认使用 `person_yolov8m-seg.pt`）
- **confidence**: 检测置信度阈值（0.0 ~ 1.0）
- **drop_area**: 最小保留面积百分比（0 ~ 99）
- **person_fullfil**: 是否生成完整人像近邻填充区域（person_area）或原始人像个体遮罩（person_masks）

#### 输出
- **mask**: 人像掩码集合

## 使用示例

将工作流example/workflowexample_everypersonseg.json拖入 ComfyUI 即可使用：

EveryPersonSegSimple Node
![工作流示例](https://github.com/CoiiChan/comfyui-every-person-seg-coii/blob/main/example/exampler_everypersonsimple.png)
EveryPersonSegSimple Node
![工作流示例](https://github.com/CoiiChan/comfyui-every-person-seg-coii/blob/main/example/exampler_everypersonsegdetail.png)

## 依赖项
- mediapipe>=0.10.13
- ultralytics>=8.0.0
- numpy>=1.21.0
- opencv-python>=4.5.0
- Pillow>=9.0.0
- torch>=1.10.0
- scipy>=1.7.0


## 模型手动下载

如果自动下载失败，请手动下载以下模型并放置到指定路径：

### YOLOv8 分割模型
- **下载地址**: https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8m-seg.pt
- **保存路径**: `ComfyUI/models/ultralytics/segm/person_yolov8m-seg.pt`

### MediaPipe 人像分割模型
- **下载地址**: https://huggingface.co/yolain/selfie_multiclass_256x256/resolve/main/selfie_multiclass_256x256.tflite?download=true
- **保存路径**: `ComfyUI/models/mediapipe/selfie_multiclass_256x256.tflite`

## 致谢
### EveryPersonSegDetail

该节点结合 YOLOv8 分割模型和 MediaPipe 人像分割模型，提供更细致的人像遮罩（支持面部、头发、身体、衣物等部位的精细分割）。遮罩轮廓优化部分代码参考了 [djbielejeski/a-person-mask-generator](https://github.com/djbielejeski/a-person-mask-generator) 项目。

#### 输入参数
- **images**: 输入图像
- **face_mask/hair_mask/body_mask/clothes_mask/background_mask**: 选择需要生成的遮罩类型
- **refine_mask**: 是否启用遮罩轮廓优化
- **confidence**: 检测置信度阈值
- **yolov_path**: YOLOv8 分割模型路径
- **drop_area**: 最小保留面积百分比

#### 输出
- **person_masks**: 精细化人像部位遮罩集合

## 致谢
本项目中的 EveryPersonSegDetail 节点遮罩轮廓优化部分代码基于 [djbielejeski/a-person-mask-generator](https://github.com/djbielejeski/a-person-mask-generator) 项目修改实现。


