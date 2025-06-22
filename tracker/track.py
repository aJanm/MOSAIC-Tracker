"""
main code for track
"""
import sys, os
import numpy as np
from scipy import stats
import torch
import cv2 
from PIL import Image
from tqdm import tqdm
import yaml 

from loguru import logger 
import argparse

from tracking_utils.envs import select_device
from tracking_utils.tools import *
from tracking_utils.visualization import plot_img, save_video
from my_timer import Timer

from tracker_dataloader import TestDataset

import sys
# sys.path.append("/data/zhangwei/code/paper-track-2/ultralytics") 
sys.path.append("/data/zoujian/packages/ultralytics")
from ultralytics import YOLO 

# trackers 
from trackers.byte_tracker import ByteTracker
from trackers.sort_tracker import SortTracker
from trackers.botsort_tracker import BotTracker
from trackers.c_biou_tracker import C_BIoUTracker
from trackers.ocsort_tracker import OCSortTracker
from trackers.deepsort_tracker import DeepSortTracker
from trackers.strongsort_tracker import StrongSortTracker
from trackers.sparse_tracker import SparseTracker
from trackers.s_iou_track import S_Ioutrack
# 2025年5月29日
from trackers.ucmc_tracker import UCMCTracker
from trackers.hybridsort_tracker import HybridSortTracker

# YOLOX modules
try:
    from yolox.exp import get_exp 
    from yolox_utils.postprocess import postprocess_yolox
    from yolox.utils import fuse_model
except Exception as e:
    logger.warning(e)
    logger.warning('Load yolox fail. If you want to use yolox, please check the installation.')
    pass 

# YOLOv7 modules
try:
    sys.path.append(os.getcwd())
    from models.experimental import attempt_load
    from utils.torch_utils import select_device, time_synchronized, TracedModel
    from utils.general import non_max_suppression, scale_coords, check_img_size
    from yolov7_utils.postprocess import postprocess as postprocess_yolov7

except Exception as e:
    logger.warning(e)
    logger.warning('Load yolov7 fail. If you want to use yolov7, please check the installation.')
    pass

# YOLOv8 modules
try:
    from ultralytics import YOLO
    from yolov8_utils.postprocess import postprocess as postprocess_yolov8

except Exception as e:
    logger.warning(e)
    logger.warning('Load yolov8 fail. If you want to use yolov8, please check the installation.')
    pass
# 2025年5月29日
# 'ucmctrack': UCMCTracker, 
# # 'hybridsort': HybridSortTracker
TRACKER_DICT = {
    'sort': SortTracker, 
    'bytetrack': ByteTracker, 
    'botsort': BotTracker, 
    'c_bioutrack': C_BIoUTracker, 
    'ocsort': OCSortTracker, 
    'deepsort': DeepSortTracker, 
    'strongsort': StrongSortTracker, 
    'sparsetrack': SparseTracker,
    's_iou_track': S_Ioutrack,
    'ucmctrack': UCMCTracker, 
    'hybridsort': HybridSortTracker
}

def get_args():
    
    parser = argparse.ArgumentParser()

    """general"""
    parser.add_argument('--dataset', type=str, default='visdrone_part', help='visdrone, mot17, etc.')
    parser.add_argument('--detector', type=str, default='yolov8', help='yolov7, yolox, etc.')
    parser.add_argument('--tracker', type=str, default='sort', help='sort, deepsort, etc')
    parser.add_argument('--reid_model', type=str, default='osnet_x0_25', help='osnet or deppsort')

    parser.add_argument('--kalman_format', type=str, default='default', help='use what kind of Kalman, sort, deepsort, byte, etc.')
    parser.add_argument('--img_size', type=int, default=1280, help='image size, [h, w]')

    # parser.add_argument('--conf_thresh', type=float, default=0.2, help='filter tracks')
    parser.add_argument('--conf_thresh', type=float, default=0.5, help='filter tracks') # 2024年12月4日
    parser.add_argument('--nms_thresh', type=float, default=0.7, help='thresh for NMS')
    parser.add_argument('--iou_thresh', type=float, default=0.5, help='IOU thresh to filter tracks')
    # parser.add_argument('--iou_thresh', type=float, default=0.3, help='IOU thresh to filter tracks')

    parser.add_argument('--device', type=str, default='6', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    """yolox"""
    parser.add_argument('--yolox_exp_file', type=str, default='./tracker/yolox_utils/yolox_m.py')

    """model path"""
    parser.add_argument('--detector_model_path', type=str, default='./weights/best.pt', help='model path')
    parser.add_argument('--trace', type=bool, default=False, help='traced model of YOLO v7')
    # other model path
    parser.add_argument('--reid_model_path', type=str, default='./weights/osnet_x0_25.pth', help='path for reid model path')
    parser.add_argument('--dhn_path', type=str, default='./weights/DHN.pth', help='path of DHN path for DeepMOT')

   
    """other options"""
    parser.add_argument('--discard_reid', action='store_true', help='discard reid model, only work in bot-sort etc. which need a reid part')
    parser.add_argument('--track_buffer', type=int, default=30, help='tracking buffer')
    parser.add_argument('--gamma', type=float, default=0.1, help='param to control fusing motion and apperance dist')
    parser.add_argument('--min_area', type=float, default=150, help='use to filter small bboxs')

    parser.add_argument('--save_dir', type=str, default='track_results/{dataset_name}_{split}_train27/')
    parser.add_argument('--exp_id', type=str, default='train33_vis_yolov8m')
    parser.add_argument('--save_images', action='store_true', help='save tracking results (image)')
    parser.add_argument('--save_videos', action='store_true', help='save tracking results (video)')
    
    parser.add_argument('--track_eval', type=bool, default=True, help='Use TrackEval to evaluate')

    return parser.parse_args()


def fuse_frames(frame1, frame2, alpha=0.5):
    """
    融合两帧图像，使用加权平均
    Args:
        frame1: 第一帧 (C, H, W)
        frame2: 第二帧 (C, H, W)
        alpha: 权重因子 (0.0 - 1.0)
    Returns:
        融合后的帧
    """
    return alpha * frame1 + (1 - alpha) * frame2



def main(args, dataset_cfgs):
    
    """1. set some params"""

    # NOTE: if save video, you must save image
    if args.save_videos:
        args.save_images = True

    """2. load detector"""
    device = select_device(args.device)

    if args.detector == 'yolox':

        exp = get_exp(args.yolox_exp_file, None)  # TODO: modify num_classes etc. for specific dataset
        model_img_size = exp.input_size
        model = exp.get_model()
        model.to(device)
        model.eval()

        logger.info(f"loading detector {args.detector} checkpoint {args.detector_model_path}")
        ckpt = torch.load(args.detector_model_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        logger.info("loaded checkpoint done")
        model = fuse_model(model)

        stride = None  # match with yolo v7

        logger.info(f'Now detector is on device {next(model.parameters()).device}')

    elif args.detector == 'yolov7':

        logger.info(f"loading detector {args.detector} checkpoint {args.detector_model_path}")
        model = attempt_load(args.detector_model_path, map_location=device)

        # get inference img size
        stride = int(model.stride.max())  # model stride
        model_img_size = check_img_size(args.img_size, s=stride)  # check img_size

        # Traced model
        model = TracedModel(model, device=device, img_size=args.img_size)
        # model.half()

        logger.info("loaded checkpoint done")

        logger.info(f'Now detector is on device {next(model.parameters()).device}')

    elif args.detector == 'yolov8':

        logger.info(f"loading detector {args.detector} checkpoint {args.detector_model_path}")
        model = YOLO(args.detector_model_path)

        model_img_size = [None, None]  
        stride = None 

        logger.info("loaded checkpoint done")

    else:
        logger.error(f"detector {args.detector} is not supprted")
        exit(0)

    """3. load sequences"""
    DATA_ROOT = dataset_cfgs['DATASET_ROOT']
    SPLIT = dataset_cfgs['SPLIT']

    seqs = sorted(os.listdir(os.path.join(DATA_ROOT, 'images', SPLIT)))
    seqs = [seq for seq in seqs if seq not in dataset_cfgs['IGNORE_SEQS']]
    if not None in dataset_cfgs['CERTAIN_SEQS']:
        seqs = dataset_cfgs['CERTAIN_SEQS']

    logger.info(f'Total {len(seqs)} seqs will be tracked: {seqs}')

    save_dir = args.save_dir.format(dataset_name=args.dataset, split=SPLIT)


    """4. Tracking"""

    # set timer 
    timer = Timer()
    seq_fps = []

    for seq in seqs:
        logger.info(f'--------------tracking seq {seq}--------------')

        dataset = TestDataset(DATA_ROOT, SPLIT, seq_name=seq, img_size=model_img_size, model=args.detector, stride=stride)

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        tracker = TRACKER_DICT[args.tracker](args, )

        process_bar = enumerate(data_loader)
        process_bar = tqdm(process_bar, total=len(data_loader), ncols=150)

        results = []

        # 初始化一个变量用于存储前一帧 2024年12月25日
        previous_frame = None
        for frame_idx, (ori_img, img) in process_bar:
            # print("for循环 img index 1:", img.shape) #for循环 img index 1: torch.Size([1, 765, 1360, 3])

            # start timing this frame
            timer.tic()

            if args.detector == 'yolov8':
                img = img.squeeze(0).cpu().numpy()

            else:
                img = img.to(device)  # (1, C, H, W)
                img = img.float() 

            ori_img = ori_img.squeeze(0)
            # 2024年12月25日
            # 检查是否存在前一帧
            if previous_frame is not None:
                              
                fused_frame = np.concatenate((previous_frame, img), axis=-1)
                # Frame 1: Fused frame shape: (765, 1360, 6)
                # print(f"Frame {frame_idx}: Fused frame shape: {fused_frame.shape}")
            else:
                # 如果前一帧不存在，仅使用当前帧
                fused_frame = img
                # Frame 0: Fused frame shape else: (765, 1360, 3)
                # print(f"Frame {frame_idx}: Fused frame shape else: {fused_frame.shape}")
                
            # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            # 更新前一帧为当前帧
            previous_frame = img


            # get detector output 
            # 
            with torch.no_grad():
                # print("track.py args.conf_thresh1:", args.conf_thresh)
                if args.detector == 'yolov8':
                    # print("track.py args.conf_thresh2:", args.conf_thresh)
                    # output = model.predict(img, conf=args.conf_thresh, iou=args.nms_thresh)
                    # print("img shape:", img.shape)
                    # print(f"img: {img.shape}, args.nms_thresh: {args.nms_thresh}")
                    # output = model.predict(img, conf=0.1, iou=args.nms_thresh) # 2024年12月11日 2024年12月25日备份

                    # 传入model.predict fused_frame.shape: (765, 1360, 6)
                    # print("传入model.predict fused_frame.shape:", fused_frame.shape)
                    output = model.predict(fused_frame, conf=0.1, iou=args.nms_thresh) # 2024年12月11日 2024年12月25日尝试改进多帧输入
                else:
                    output = model(img)

            # postprocess output to original scales
            if args.detector == 'yolox':
                output = postprocess_yolox(output, len(dataset_cfgs['CATEGORY_NAMES']), conf_thresh=args.conf_thresh, 
                                           img=img, ori_img=ori_img)

            elif args.detector == 'yolov7':
                output = postprocess_yolov7(output, args.conf_thresh, args.nms_thresh, img.shape[2:], ori_img.shape)

            elif args.detector == 'yolov8':
                output = postprocess_yolov8(output)
            
            else: raise NotImplementedError

            # output: (tlbr, conf, cls)
            # convert tlbr to tlwh
            if isinstance(output, torch.Tensor): 
                output = output.detach().cpu().numpy()
            output[:, 2] -= output[:, 0]
            output[:, 3] -= output[:, 1]
            # print("shape of output, track.py: ", output.shape, "type:", type(output))
            # 一行计算所有统计值
            # median, mode_result, min_val, max_val, q1, q3, std_dev, mean_val = np.median(output), stats.mode(output), np.min(output), np.max(output), np.percentile(output, 25), np.percentile(output, 75), np.std(output), np.mean(output)

            # 打印所有结果
            # print(f"中位数: {median}, 众数: {mode_result.mode[0]}, 出现次数: {mode_result.count[0]}, 值的分布范围: [{min_val}, {max_val}], 第一四分位数 (Q1): {q1}, 第三四分位数 (Q3): {q3}, 标准差: {std_dev}, 均值: {mean_val}")
            # print("output:", output)
            current_tracks = tracker.update(output, img, ori_img.cpu().numpy())
        
            # save results
            cur_tlwh, cur_id, cur_cls, cur_score = [], [], [], []
            for trk in current_tracks:
                bbox = trk.tlwh
                id = trk.track_id
                cls = trk.category
                score = trk.score

                # filter low area bbox
                if bbox[2] * bbox[3] > args.min_area:
                    cur_tlwh.append(bbox)
                    cur_id.append(id)
                    cur_cls.append(cls)
                    cur_score.append(score)
                    # results.append((frame_id + 1, id, bbox, cls))

            results.append((frame_idx + 1, cur_id, cur_tlwh, cur_cls, cur_score))

            timer.toc()

            # 2025年1月17日注释
            # if args.save_images:
            #     plot_img(img=ori_img, frame_id=frame_idx, results=[cur_tlwh, cur_id, cur_cls], 
            #              save_dir=os.path.join(save_dir, 'vis_results'))
                
            # 2025年1月17日 todo修改保存路径，再加上绘制图片，注意这里内容会很大
            # print("args.save_images:", args.save_images)
            # if args.save_images:
            tmp_save = True # 此时暂时将其关闭 2025年5月27日 2025年6月2日再打开保存一份botsort结果
            if tmp_save:   # visdrone保存 2025年1月17日上午
                plot_img(img=ori_img, frame_id=frame_idx, results=[cur_tlwh, cur_id, cur_cls],
                # save_dir=os.path.join("/data/zoujian/Detect_tracking/my_trackingresults/", args.exp_id, 'track_vis_results', seq))
                save_dir=os.path.join("/data/zoujian/Detect_tracking/my_trackingresults/", args.exp_id, 'track_vis_results526', seq)) # 2025年5月26日 
            # if tmp_save:   # uavdt保存 2025年1月17日下午 后面想想还是在zw服务器跑uavdt
            #     plot_img(img=ori_img, frame_id=frame_idx, results=[cur_tlwh, cur_id, cur_cls],
            #     save_dir=os.path.join("/data/zoujian/Detect_tracking/my_trackingresults/", args.exp_id, 'track_uavdt_results', seq))

        # save_results(folder_name=os.path.join(args.dataset, SPLIT), 
        #              seq_name=seq, 
        #              results=results)
        exp_id = args.exp_id
        save_results(folder_name=os.path.join(exp_id, args.dataset, SPLIT), 
                     seq_name=seq, 
                     results=results)
        
        
        
        # show the fps
        seq_fps.append(frame_idx / timer.total_time)
        logger.info(f'fps of seq {seq}: {seq_fps[-1]}')
        timer.clear()
        
        if args.save_videos:
            save_video(images_path=os.path.join(save_dir, 'vis_results'))
            logger.info(f'save video of {seq} done')

    # show the average fps
    logger.info(f'average fps: {np.mean(seq_fps)}')


if __name__ == '__main__':

    args = get_args()

    with open(f'./tracker/config_files/{args.dataset}.yaml', 'r') as f:
        cfgs = yaml.load(f, Loader=yaml.FullLoader)

        
    main(args, cfgs)
