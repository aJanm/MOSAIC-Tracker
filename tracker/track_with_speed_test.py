"""
main code for track
"""
# 1.在文件头部添加
# import time
# from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
# import GPUtil


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


# 设备初始化代码后添加

# def print_gpu_usage():
#     gpus = GPUtil.getGPUs()
#     for gpu in gpus:
#         print(f"GPU {gpu.id}: Load {gpu.load*100:.1f}% | Mem Used {gpu.memoryUsed}MB/{gpu.memoryTotal}MB")

# 1.添加在设备初始化之后
import time
import torch
from GPUtil import showUtilization as gpu_usage  # 需先安装gputil

from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, \
                   nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo

# def init_gpu_monitor():
#     torch.cuda.empty_cache()
#     torch.cuda.reset_max_memory_allocated()
#     print("GPU初始化状态:")
#     gpu_usage()
#     return time.time()

# start_total_time = init_gpu_monitor()
# total_frames = 0
# inference_time_acc = 0.0
# tracking_time_acc = 0.0



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

    """1. 初始化性能监控工具""" 
    # 总时间计时器
    # total_start_time = time.time()
    # GPU显存监控初始化
    # 2025年5月29日
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(3)  # 监控第3个物理gpu

    

    # 用于统计每帧的检测和跟踪耗时
    # det_times   = []
    # track_times = []
    
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
    # # 全局帧计时器，用于算 FPS
    # set timer 
    timer = Timer()
    seq_fps = []

    # 准备两个列表，用来存储每次采样的指标 2025年5月29日
    all_gpu_utils = []
    all_gpu_mems  = []
     # 全局统计用，不清空
    all_det_times       = []
    all_track_times     = []

    for seq in seqs:
        logger.info(f'--------------tracking seq {seq}--------------')

        dataset = TestDataset(DATA_ROOT, SPLIT, seq_name=seq, img_size=model_img_size, model=args.detector, stride=stride)

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        tracker = TRACKER_DICT[args.tracker](args, )

        process_bar = enumerate(data_loader)
        process_bar = tqdm(process_bar, total=len(data_loader), ncols=150)

        results = []


        # 用于统计每个序列内部的检测/跟踪耗时 2025年5月29日
        per_seq_det_times   = []
        per_seq_track_times = []
        per_seq_utils, per_seq_mems = [], []

        # 初始化一个变量用于存储前一帧 2024年12月25日
        previous_frame = None
        for frame_idx, (ori_img, img) in process_bar:
            # print("for循环 img index 1:", img.shape) #for循环 img index 1: torch.Size([1, 765, 1360, 3])

            # 2025年5月29日 
            # 在每帧开始或结束时采样一次，比如放在 timer.toc() 之后
            # 你也可以每处理 N 帧采一次：
            N = 10
            if frame_idx % N == 0:
                u = nvmlDeviceGetUtilizationRates(handle)
                m = nvmlDeviceGetMemoryInfo(handle)
                per_seq_utils.append(u.gpu)               # GPU 利用率（%）
                per_seq_mems.append(m.used // 1024**2)    # 显存占用（MiB）

                # 同时，也把这次采样，推到全局列表里
                all_gpu_utils.append(u.gpu)
                all_gpu_mems.append(m.used // 1024**2)

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
                # 将当前帧与前一帧进行融合
                # fused_frame = fuse_frames(previous_frame, img)  # 融合函数需定义
                # fused_frame = previous_frame + img
                # 2024年12月26日
                # 将当前帧和前一帧在通道维度上拼接 (C, H, W -> 2*C, H, W)
                # fused_frame = torch.cat((previous_frame, img), dim=-1)

                # if previous_frame is not None: (765, 1360, 3) img: (765, 1360, 3)
                # print("if previous_frame is not None:", previous_frame.shape, "img:", img.shape) 

                
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

            # 在主循环中定期调用 2025年5月29日 计算GPU占用
            # if frame_idx % 10 == 0:
            #     print_gpu_usage()

            # get detector output 2025年5月29日
            # —— 检测计时 —— 
            det_start = time.time()
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

            # 2025年5月29日
            det_end = time.time()
            det_t = det_end - det_start
            per_seq_det_times.append(det_t)
            all_det_times.append(det_t)

            print("shape of output, track.py: ", output.shape, "type:", type(output))
            # 一行计算所有统计值
            # median, mode_result, min_val, max_val, q1, q3, std_dev, mean_val = np.median(output), stats.mode(output), np.min(output), np.max(output), np.percentile(output, 25), np.percentile(output, 75), np.std(output), np.mean(output)

            # 打印所有结果
            # print(f"中位数: {median}, 众数: {mode_result.mode[0]}, 出现次数: {mode_result.count[0]}, 值的分布范围: [{min_val}, {max_val}], 第一四分位数 (Q1): {q1}, 第三四分位数 (Q3): {q3}, 标准差: {std_dev}, 均值: {mean_val}")
            # print("output:", output)
            
            # 2. # --- 跟踪阶段监控 ---
            # my_tracking_start = time.time()
            
            # —— 跟踪计时 —— 2025年5月29日
            track_start = time.time()

            current_tracks = tracker.update(output, img, ori_img.cpu().numpy())

            # torch.cuda.synchronize()
            # my_tracking_time = time.time() - my_tracking_start
            # my_tracking_time_acc += my_tracking_time

            # 2025年5月29日
            track_end = time.time()
            tr_t = track_end - track_start
            per_seq_track_times.append(tr_t)
            all_track_times.append(tr_t)
                    
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

            timer.toc() # 在每帧处理结束时计算时间差并累加总时间

            # 2025年1月17日注释
            # if args.save_images:
            #     plot_img(img=ori_img, frame_id=frame_idx, results=[cur_tlwh, cur_id, cur_cls], 
            #              save_dir=os.path.join(save_dir, 'vis_results'))
                
            # 2025年1月17日 todo修改保存路径，再加上绘制图片，注意这里内容会很大
            # print("args.save_images:", args.save_images)
            # if args.save_images:
            tmp_save = False # 此时暂时将其关闭 2025年5月27日
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
        
        
        # GPU 记录完一个序列之后，可以临时打印一下这一序列的统计,2025年5月29日
        seq_avg_util = sum(per_seq_utils) / len(per_seq_utils)
        seq_max_mem  = max(per_seq_mems)
        logger.info(f"[{seq}] 平均 GPU 利用率: {seq_avg_util:.1f}%  显存峰值: {seq_max_mem} MiB")

        # 然后如果你想把列表清空，进入下一个序列重新采样：
        per_seq_utils.clear()
        per_seq_mems.clear()

        
        # show the fps
        seq_fps.append(frame_idx / timer.total_time)
        logger.info(f'fps of seq {seq}: {seq_fps[-1]}')
        timer.clear()


        # 序列结束，输出这一序列的检测／跟踪统计 2025年5月29日
        if per_seq_det_times:
            avg_det = sum(per_seq_det_times) / len(per_seq_det_times)
            max_det = max(per_seq_det_times)
            avg_tr  = sum(per_seq_track_times) / len(per_seq_track_times)
            max_tr  = max(per_seq_track_times)
            logger.info(
                f"[{seq}] 检测 avg: {avg_det*1000:.1f}ms  max: {max_det*1000:.1f}ms | "
                f"跟踪 avg: {avg_tr*1000:.1f}ms  max: {max_tr*1000:.1f}ms"
            )

        # 清空序列列表，为下一个序列做准备
        per_seq_det_times.clear()
        per_seq_track_times.clear()
        
        if args.save_videos:
            save_video(images_path=os.path.join(save_dir, 'vis_results'))
            logger.info(f'save video of {seq} done')

    
    # 全部序列跑完后，还可以汇总所有数据：
    # (如果上面每个序列都 clear，就得在外面再用另外两个全局列表all_gpu_utils,all_gpu_mems采样)
    overall_avg_util = sum(all_gpu_utils) / len(all_gpu_utils)
    overall_max_mem  = max(all_gpu_mems)
    logger.info(f"全部序列 —— 平均 GPU 利用率: {overall_avg_util:.1f}%, 显存峰值: {overall_max_mem} MiB")

    # show the average fps
    logger.info(f'average fps: {np.mean(seq_fps)}')

    # 全序列检测／跟踪统计
    if all_det_times:
        overall_avg_det = sum(all_det_times) / len(all_det_times)
        overall_max_det = max(all_det_times)
        overall_avg_tr  = sum(all_track_times) / len(all_track_times)
        overall_max_tr  = max(all_track_times)
        logger.info(
            f"所有序列 —— 检测 avg: {overall_avg_det*1000:.1f}ms  max: {overall_max_det*1000:.1f}ms | "
            f"跟踪 avg: {overall_avg_tr*1000:.1f}ms  max: {overall_max_tr*1000:.1f}ms"
        )




if __name__ == '__main__':

    args = get_args()

    with open(f'./tracker/config_files/{args.dataset}.yaml', 'r') as f:
        cfgs = yaml.load(f, Loader=yaml.FullLoader)

        
    main(args, cfgs)
