# MOSAIC-Tracker: Mutual-enhanced Occlusion-aware Spatiotemporal Adaptive Identity Consistency Network for Aerial Multi-Object Tracking

This repository provides the implementation for paper.  We present a Mutual-enhanced Occlusion-aware Spatiotemporal Adaptive  Identity Conservation Network (MOSAIC-Tracker) that dynamically  optimizes feature representation and data association across three key  dimensions. Extensive evaluations on UAVDT and VisDrone2019 datasets  demonstrate good performance.

1.To enhance reproducibility and facilitate implementation, we have additionally provided **our modified version of the ultralytics code package** as supplementary material. Users may directly reference and integrate these customized files into their workflows, rather than relying solely on the official ultralytics distribution.

2.**Easier_To_Use_The TrackEval** code package is used to calculate metrics such as HOTA,MOTA, and IDF1 in multi-object tracking. The official link to this package is: [Easier_To_Use_TrackEval](https://github.com/JackWoo0831/Easier_To_Use_TrackEval). The evaluation can be performed using **either the official metric computation toolkit or the package contained within our file directory.**

## 🛠️ Installation

> ⚠️ **Note:** The provided environment file is for **reference only**. Some packages may require manual installation or configuration.

### Recommended Environment

- `pytorch` 2.4.1 

- `python` 3.9 or 3.10  

  > ⚙️ You can also run it on older versions of PyTorch first to test if relevant performance results can be obtained. [installation instructions](https://pytorch.org/get-started/locally/).  

- `opencv-python` 4.10.0

- `timm` 1.0.12  

- `wandb` (for experiment tracking)

- `torchvision ` 0.19.1  

- `pyside6` 6.8.1

- `matplotlib` 3.9.4

---

## 🚀 Usage

### ⚙️ Image Configuration File

1. Navigate to `./tracker/config_files`, locate the corresponding configuration file and modify it.
2. Go to the `data` folder, then:
   - Edit the YAML configuration file
   - Update the file indices in the YAML to point to the correct text files (train.txt and test.txt)
   - Replace the content in these text files with your custom file paths

---

### 📂 Data Preparation

Organize your dataset in the following structure under `data_dir`:

```
📂 UAVDT2
    📂 images # Uav images
    	📂 test
    	📂 train
    📂 labels 
    📂 UAV-benchmark-M
    📂 UAV-benchmark-MOTD_v1.0
    
📂 VisDrone-MOT2
    📂 annotations
    📂 annotations_distmot
    📂 images
        📂VisDrone2019-MOT-test-dev
        📂VisDrone2019-MOT-train
    📂 labels
    📂 labels with ids
    📂 merge_cls_gt
```
The download URLs for both datasets are presented below. Upon data acquisition, please organize the files in their designated directories as per the specified data hierarchy.
- [VisDrone dataset](https://github.com/VisDrone/VisDrone-Dataset)
- [UAVDT dataset](https://sites.google.com/view/grli-uavdt/)
---

### 🎯 Training
Due to variations in server configurations, the optimal number of training epochs may differ across users. It is recommended to perform comparative experiments and retain the model with the highest validation performance.
To start training with the provided example configuraton, simply run

```bash
python tracker/yolov8_utils/train_volov8.py --epochs=10 --device=0
python tracker/yolov8_utils/train_yolov8_uavdt.py --epochs=10 --device=0
```

After training, there will be several checkpoint files under the  directory. It supports parallel training.

---

### 📊 Evaluation

Evaluate a trained model with:

```bash
python tracker/track.py --dataset visdrone --detector yolov8 --tracker s_iou_track --kalman_format bot --detector_model_path "./best.pt" --exp_id train_test_vis_yolov8s --device=3 --conf_thresh=0.4 && cd ./Easier_To_Use_TrackEval && python scripts/run_custom_dataset.py --config_path configs/VisDrone_test_dev.yaml


python tracker/track.py --dataset uavdt --detector yolov8 --tracker s_iou_track --kalman_format bot --detector_model_path "./best.pt" --exp_id train_test_uav_yolov8s --device=0 --conf_thresh=0.4 && cd /data/zoujian/Easier_To_Use_TrackEval && python scripts/run_custom_dataset.py --config_path configs/UAVDT_test.yaml

```

First, activate your environment, then go to the Yolov7-tracker-2/ directory and run the track.py script inside the tracker folder.  Replace pt file with any others as needed. Replace `detector_model_path` and `Easier_To_Use_TrackEval` with your custom paths. We will publish the pt files in the VisDrone and UAVDT datasets.  

The main results of VisDrone:

| **Method** | **Dataset** | **HOTA** | **MOTA** | **IDF1** | **FN** | **FP** | **IDs** | **MT** | **ML** |                                           **URL**                                           |
|:----------:|:-----------:|:--------:|:--------:|:--------:|:---------:|:---------:|:---------:|-----------|----------| :-----------------------------------------------------------------------------------------: |
| Ours  | VisDrone |46.044 | 41.358 | 58.142 | 126112 | 26895 | 1862 | 583 | 591 | [model](https://github.com/aJanm/MOSAIC-Tracker/releases/download/v1.0/my_visdrone.pt) |

| **Method** | **Dataset** | **HOTA** | **MOTA** | **IDF1** | **FN** | **FP** | **IDs** | **MT** | **ML** |                                           **URL**                                           |
|:----------:|:-----------:|:--------:|:--------:|:--------:|:---------:|:---------:|:---------:|-----------|----------| :-----------------------------------------------------------------------------------------: |
| Ours  | UAVDT | 64.568 | 70.294 | 81.667 | 19584 | 3137 | 375 | 190 | 38 | [model](https://github.com/aJanm/MOSAIC-Tracker/releases/download/v1.0/my_uavdt.pt) |

### Acknowledgements
We're grateful to the open-source authors whose code has assisted us.
- [Yolov7-tracker](https://github.com/JackWoo0831/Yolov7-tracker/tree/v2)
- [Easier_To_Use_TrackEval](https://github.com/JackWoo0831/Easier_To_Use_TrackEval)
- [Ultralytics](https://github.com/ultralytics/ultralytics)

