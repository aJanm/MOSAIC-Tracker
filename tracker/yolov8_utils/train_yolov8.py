import torch 

import sys
# sys.path.append("/data/zhangwei/code/paper-track-2/ultralytics") 
sys.path.append("/data/zoujian/packages/ultralytics") 
from ultralytics import YOLO 
import numpy as np  

import argparse

import os
import wandb
os.environ["WANDB_API_KEY"] = 'KEY'
os.environ["WANDB_MODE"] = "offline"


def main(args):
    """ main func
    
    """
    # model = YOLO("yolov8x.yaml")  # build a new model from scratch
    # model = YOLO(model=args.model_weight)
    # /data/zhangwei/code/paper-track-2/Yolov7-tracker-2/weights/yolov8m.pt
    # /data/zhangwei/code/paper-track-2/ultralytics/ultralytics/cfg/models/mamba_tolo.yaml
    # /data/zoujian/Detect_tracking/Yolov7-tracker-2/weights/yolov8s.pt

    model = YOLO('/data/zoujian/Detect_tracking/Yolov7-tracker-2/weights/yolov8s.pt')  # pass any model type  2025年1月6日

    # model = YOLO('/data/zoujian/Detect_tracking/Yolov7-tracker-2/weights/yolov8m.pt')
    # model = YOLO('/data/zoujian/Detect_tracking/Yolov7-tracker-2/weights/yolov8n.pt') # 2024年11月28日
    # model = YOLO('/data/zoujian/packages/ultralytics/ultralytics/cfg/models/v8/yolov8_CAFM.yaml') # 2024年11月28日
    # model = YOLO('/data/zoujian/packages/ultralytics/ultralytics/cfg/models/v8/yolov8m_CAFM.yaml') # 2024年11月28日
    # model = YOLO('/data/zoujian/packages/ultralytics/ultralytics/cfg/models/v8/yolov8_CMUNeXt.yaml') # 2024年11月28日
    # model = YOLO('/data/zoujian/packages/ultralytics/ultralytics/cfg/models/v8/yolov8m_CMUNeXt.yaml') # 2024年11月28日
    # model = YOLO('/data/zoujian/packages/ultralytics/ultralytics/cfg/models/v8/yolov8s_CMUNeXt.yaml') # 2024年12月3日

    # model = YOLO('/data/zoujian/packages/ultralytics/ultralytics/cfg/models/v8/yolo8n_iRMB.yaml') # 2024年11月29日
    # model = YOLO('/data/zoujian/packages/ultralytics/ultralytics/cfg/models/v8/yolo8m_iRMB.yaml') # 2024年11月29日
    # model = YOLO('/data/zoujian/packages/ultralytics/ultralytics/cfg/models/v8/yolo8s_iRMB.yaml') # 2024年12月2日
    # model = YOLO('/data/zoujian/packages/ultralytics/ultralytics/cfg/models/v8/yolo8n_C2f_FSDA.yaml') # 2024年11月29日
    # model = YOLO('/data/zoujian/packages/ultralytics/ultralytics/cfg/models/v8/yolo8n_PKInet.yaml') # 2024年11月29日
    # model = YOLO('/data/zoujian/packages/ultralytics/ultralytics/cfg/models/v8/yolo8n_C2f_OAttention.yaml') # 2024年11月29日
    # model = YOLO('/data/zoujian/packages/ultralytics/ultralytics/cfg/models/v8/yolov8-p2.yaml') # 2024年11月30日
    # model = YOLO('/data/zoujian/packages/ultralytics/ultralytics/cfg/models/v8/yolov8s_smallhead.yaml') # 2024年12月3日
    # model = YOLO('/data/zoujian/packages/ultralytics/ultralytics/cfg/models/v8/yolo8s_seam2.yaml') # 2024年12月3日
    # model = YOLO('/data/zoujian/packages/ultralytics/ultralytics/cfg/models/v8/yolov8s.yaml') # 2024年12月3日   2024年12月24日
    # model = YOLO('/data/zoujian/packages/ultralytics/ultralytics/cfg/models/v8/yolo8s_MultiSeam.yaml') # 2024年12月3日
    # model = YOLO('/data/zoujian/packages/ultralytics/ultralytics/cfg/models/v8/yolo8s_MSCAM.yaml') # 2024年11月28日 2024年12月26日 2025年1月6日

    # model.load("/data/zoujian/Detect_tracking/Yolov7-tracker-2/weights/yolov8m.pt")
    # model.load("/data/zoujian/Detect_tracking/Yolov7-tracker-2/weights/yolov8n.pt")
    model.load("/data/zoujian/Detect_tracking/Yolov7-tracker-2/weights/yolov8s.pt")
    model.train(
        data=args.data_cfg,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_sz,
        patience=50,  # epochs to wait for no observable improvement for early stopping of training
        pretrained=True,
        device=args.device, # 2024年11月28日
        amp=False,
        overlap_mask=False,
        hsv_h=0,
        hsv_s=0,
        hsv_v=0,
        weight_decay=0.0001,
        # device=[0,1,2,3] # 多卡训练
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser("YOLO v8 train parser")
    
    parser.add_argument('--model', type=str, default='yolov8m.yaml', help='yaml or pt file')
    parser.add_argument('--model_weight', type=str, default='/data/zhangwei/code/paper-track-2/Yolov7-tracker-2/weights/yolov8m.pt', help='')
    # parser.add_argument('--data_cfg', type=str, default='/data/zhangwei/code/paper-track-2/Yolov7-tracker-2/tracker/yolov8_utils/data_cfgs/visdrone.yaml', help='')
    parser.add_argument('--data_cfg', type=str, default='/data/zoujian/Detect_tracking/Yolov7-tracker-2/tracker/yolov8_utils/data_cfgs/visdrone.yaml', help='')
    parser.add_argument('--epochs', type=int, default=20, help='')
    # parser.add_argument('--batch_size', type=int, default=8, help='') # m 16 x 4  # 2024年12月7日 原始yolov8s
    parser.add_argument('--batch_size', type=int, default=4, help='') # m 16 x 4 # 2024年12月8日
    parser.add_argument('--img_sz', type=int, default=1280, help='')
    parser.add_argument('--device', type=str, default='2', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    args = parser.parse_args()

    main(args)