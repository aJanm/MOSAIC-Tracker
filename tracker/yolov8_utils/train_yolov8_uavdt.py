import torch 
import sys
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
    # model = YOLO('/data/zhangwei/code/paper-track-2/Yolov7-tracker-2/weights/yolov8n.pt')  # pass any model type
    model = YOLO('/data/zoujian/packages/ultralytics/ultralytics/cfg/models/v8/yolov8s.yaml') # 2024年12月12日
    # model = YOLO('/data/zoujian/packages/ultralytics/runs/detect/train115/weights/last.pt') # 2024年12月20日 基于得到的训练10轮后，接着训练
    # model = YOLO('/data/zoujian/packages/ultralytics/runs/detect/train115/weights/best.pt') # 2024年12月20日 基于得到的训练best 10轮后，接着训练
    # model = YOLO('/data/zoujian/packages/ultralytics/ultralytics/cfg/models/v8/yolo8s_seam2.yaml') # 2024年12月14日
    # model = YOLO('/data/zoujian/packages/ultralytics/ultralytics/cfg/models/v8/yolo8s_seam.yaml') # 2024年12月20日
    # model = YOLO('/data/zoujian/packages/ultralytics/ultralytics/cfg/models/v8/yolo8s_MSCAM.yaml') # 2024年12月19日
    # model = YOLO('/data/zoujian/packages/ultralytics/runs/detect/train148/weights/last.pt') # 2024年12月20日 yolo8s_MSCAM 基于得到的10轮，后面接着训练
    # model = YOLO('/data/zoujian/packages/ultralytics/ultralytics/cfg/models/v8/yolov8s_CAFM.yaml') # 2024年12月19日
    # model.load("/data/zoujian/Detect_tracking/Yolov7-tracker-2/weights/yolov8s.pt")
    model.train(
        data=args.data_cfg,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_sz,
        patience=50,  # epochs to wait for no observable improvement for early stopping of training
        pretrained=True,
        # device=args.device, # 2024年12月19日
        device="cuda:2",
        amp=False, #2024年12月14日 device无效 尝试加一下以下这些
        overlap_mask=False,
        hsv_h=0,
        hsv_s=0,
        hsv_v=0,
        translate=0,
        scale=0,
        fliplr=0, 
        mixup=0,
        copy_paste=0,
        flipud=0,
        degrees=0,
        erasing=0, # 2024年12月23日
        

    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser("YOLO v8 train parser")
    
    parser.add_argument('--model', type=str, default='yolov8n.yaml', help='yaml or pt file')
    parser.add_argument('--model_weight', type=str, default='/data/zhangwei/code/paper-track-2/Yolov7-tracker-2/weights/yolov8n.pt', help='')
    # parser.add_argument('--data_cfg', type=str, default='/data/zhangwei/code/paper-track-2/Yolov7-tracker-2/tracker/yolov8_utils/data_cfgs/uavdt.yaml', help='')
    parser.add_argument('--data_cfg', type=str, default='/data/zoujian/Detect_tracking/Yolov7-tracker-2/tracker/yolov8_utils/data_cfgs/uavdt.yaml', help='') # 2024年12月12日
    parser.add_argument('--epochs', type=int, default=20, help='')
    parser.add_argument('--batch_size', type=int, default=6, help='')
    parser.add_argument('--img_sz', type=int, default=1280, help='')
    parser.add_argument('--device', type=str, default='2', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    args = parser.parse_args()
    print("device:",args.device)

    main(args)