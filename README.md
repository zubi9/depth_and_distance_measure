# Depth and Distance Measure

## Description

**Monocular Depth Estimation:** is the task of estimating the depth value (distance relative to the camera) of each pixel given a single (monocular) RGB image. This challenging task is a key prerequisite for determining scene understanding for applications such as 3D scene reconstruction, autonomous driving, and AR.

#### What is Distance Calculation?

Measuring the gap between two objects is known as distance calculation within a specified space. In the case of Ultralytics YOLOv8, the bounding box centroid is employed to calculate the distance for bounding boxes highlighted by the user.

##### Advantages of Distance Calculation

- **Localization Precision**: Enhances accurate spatial positioning in computer vision tasks.
- **Size Estimation**: Allows estimation of physical sizes for better contextual understanding.
- **Scene Understanding**: Contributes to a 3D understanding of the environment for improved decision-making.

## Usage

<p align="center">
    <img src="assets/Screenshot from 2024-07-03 15-45-04.png" alt="Depth estimation and object detection" width="600">
</p>

<p align="center">
    <img src="assets/Screenshot from 2024-07-06 01-24-59.png" alt="Depth map gradion app" width="600">
</p>

### Prerequisites

#### Pre-trained Models

Download one of the provided **four models** of varying scales for robust relative depth estimation and keep in `checkpoints` directory:

| Model | Params | Checkpoint |
|:-|-:|:-:|
| Depth-Anything-V2-Small | 24.8M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true) |
| Depth-Anything-V2-Base | 97.5M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true) |
| Depth-Anything-V2-Large | 335.3M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true) |
| Depth-Anything-V2-Giant | 1.3B | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Giant/resolve/main/depth_anything_v2_vitg.pth?download=true) |

### Installation

```bash
git clone https://github.com/zubi9/depth_and_distance_measure.git
cd depth_and_distance_measure
pip install -r requirements.txt
```

### Running

#### Live Depth and Distance Measurement

To run the script for side-by-side YOLO v8 distance measurement and monocular depth estimation with a webcam:

```bash
python dnd_live_only.py
```

#### Depth and Distance Estimation on Images

```bash
python run.py --encoder <vits | vitb | vitl | vitg> --img-path <path> --outdir <outdir> [--input-size <size>] [--pred-only] [--grayscale]
```

Options:
- `--encoder`: Choose from `vits`, `vitb`, `vitl`, `vitg`.
- `--img-path`: Path to an image directory, single image, or a text file with image paths.
- `--input-size` (optional): Default is `518`. Increase for finer results.
- `--pred-only` (optional): Only save the predicted depth map.
- `--grayscale` (optional): Save the grayscale depth map without applying a color palette.

Example:
```bash
python run.py --encoder vitg --img-path assets/examples --outdir depth_vis
```

#### Depth Estimation on Videos

```bash
python run_video.py --encoder vitg --video-path assets/examples_video --outdir video_depth_vis
```

*Note: The larger models provide better temporal consistency on videos.*

### Gradio Demo

To use the Gradio demo locally:

```bash
python app.py
```

## Citation

If you find this project useful, please consider citing:

```bibtex
@article{depth_anything_v2,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal={arXiv preprint arXiv:},
  year={2024}
}
```

## References

- Yolo v8: [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)

## Linces
-  [Apache License](LICENSE)
-  [GNU License]
