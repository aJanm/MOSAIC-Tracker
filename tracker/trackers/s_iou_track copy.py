"""
Bot sort
"""

import numpy as np  
import torch 
from torchvision.ops import nms

import cv2 
import torchvision.transforms as T

from .basetrack import BaseTrack, TrackState
from .tracklet import Tracklet, Tracklet_w_reid # 表示一个短暂的跟踪片段，通常是目标在某段时间内的连续检测结果。 Tracklet_w_reid：可能是一个扩展版本，包含重新识别（re-identification）功能，用于在目标消失后重新确认其身份。
from .matching import * # 导入可能包含与目标匹配相关的各种函数和类，例如通过计算相似度或距离来将当前检测到的目标与先前的目标进行关联

from .reid_models.OSNet import *
from .reid_models.load_model_tools import load_pretrained_weights
from .reid_models.deepsort_reid import Extractor

from .camera_motion_compensation import GMC

REID_MODEL_DICT = {
    'osnet_x1_0': osnet_x1_0, 
    'osnet_x0_75': osnet_x0_75, 
    'osnet_x0_5': osnet_x0_5, 
    'osnet_x0_25': osnet_x0_25, 
    'deepsort': Extractor
}


def load_reid_model(reid_model, reid_model_path):
    
    if 'osnet' in reid_model:
        func = REID_MODEL_DICT[reid_model]
        model = func(num_classes=1, pretrained=False, )
        load_pretrained_weights(model, reid_model_path)
        model.cuda().eval()
        
    elif 'deepsort' in reid_model:
        model = REID_MODEL_DICT[reid_model](reid_model_path, use_cuda=True)

    else:
        raise NotImplementedError
    
    return model

class S_Ioutrack(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_tracklets = []  # type:  当前正在跟踪的轨迹列表。
        self.lost_tracklets = []  # type: list[Tracklet] 丢失的轨迹列表，可能由于对象消失或未检测到
        self.removed_tracklets = []  # type: list[Tracklet] 被移除的轨迹列表，通常是由于超出跟踪时间或其他条件。

        self.frame_id = 0
        self.args = args

        self.det_thresh = args.conf_thresh + 0.1 # 检测阈值，通过将传入的置信度阈值加0.1来设置。
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size

        self.motion = args.kalman_format

        self.with_reid = not args.discard_reid # 是否使用再识别模型，取决于传入参数。

        self.reid_model, self.crop_transforms = None, None 
        if self.with_reid:
            self.reid_model = load_reid_model(args.reid_model, args.reid_model_path)
            self.crop_transforms = T.Compose([
            # T.ToPILImage(),
            # T.Resize(size=(256, 128)),
            T.ToTensor(),  # (c, 128, 256)
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
            

        # camera motion compensation module  创建一个摄像机运动补偿对象，使用 ORB 方法，降低图像分辨率（downscale=2）
        self.gmc = GMC(method='orb', downscale=2, verbose=None)

    def reid_preprocess(self, obj_bbox):
        """
        preprocess cropped object bboxes 
        
        obj_bbox: np.ndarray, shape=(h_obj, w_obj, c)

        return: 
        torch.Tensor of shape (c, 128, 256)
        """
        # 将输入的 obj_bbox 转换为浮点数并归一化（像素值范围在0到1之间），然后调整大小为 (128, 128)。
        obj_bbox = cv2.resize(obj_bbox.astype(np.float32) / 255.0, dsize=(128, 128))  # shape: (128, 256, c)

        return self.crop_transforms(obj_bbox)

    '''
    从给定的边界框和原始图像中提取对象的外观特征。它首先对每个对象的图像进行裁剪和预处理，然后通过再识别模型获取特征，
    最后返回这些特征的 NumPy 数组。这在多目标跟踪和再识别任务中是一个重要步骤，有助于在不同帧之间识别和跟踪相同的对象。
    '''
    def get_feature(self, tlwhs, ori_img):
        """
        get apperance feature of an object
        tlwhs: shape (num_of_objects, 4)
        ori_img: original image, np.ndarray, shape(H, W, C)
        """
        obj_bbox = []

        for tlwh in tlwhs:
            tlwh = list(map(int, tlwh))
            # if any(tlbr_ == -1 for tlbr_ in tlwh):
            #     print(tlwh)
            #  一个形状为 (num_of_objects, 4) 的数组 调用 reid_preprocess 方法对裁剪的图像进行预处理，返回的张量被添加到 obj_bbox 列表中。
            tlbr_tensor = self.reid_preprocess(ori_img[tlwh[1]: tlwh[1] + tlwh[3], tlwh[0]: tlwh[0] + tlwh[2]])
            obj_bbox.append(tlbr_tensor)
        
        if not obj_bbox:
            return np.array([])
        
        obj_bbox = torch.stack(obj_bbox, dim=0)
        obj_bbox = obj_bbox.cuda()  
        
        features = self.reid_model(obj_bbox)  # shape: (num_of_objects, feature_dim) 将预处理后的对象张量传递给再识别模型 self.reid_model，提取每个对象的特征。
        return features.cpu().detach().numpy()


    # 用于更新跟踪器的状态，处理检测结果并管理跟踪目标的状态。
    def update(self, output_results, img, ori_img):
        """
        output_results: processed detections (scale to original size) tlwh format
        """

        self.frame_id += 1
        activated_tracklets = []
        refind_tracklets = []
        lost_tracklets = []
        removed_tracklets = []
        # 从输出结果中提取置信度分数、边界框和类别信息。
        scores = output_results[:, 4]
        bboxes = output_results[:, :4]
        categories = output_results[:, -1]

        # 筛选检测结果,根据置信度分数筛选检测框：remain_inds 表示高于阈值的检测框。inds_second 用于表示低置信度的检测框。
        remain_inds = scores > self.args.conf_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.conf_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]

        # 处理类别和分数 分别获取高置信度和低置信度检测框的类别和分数。
        cates = categories[remain_inds]
        cates_second = categories[inds_second]
        
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        """Step 1: Extract reid features"""
        if self.with_reid:
            features_keep = self.get_feature(tlwhs=dets[:, :4], ori_img=ori_img)

        # 创建检测对象 根据检测框和特征创建相应的跟踪目标（Tracklet 或 Tracklet_w_reid）。
        if len(dets) > 0:
            if self.with_reid:
                detections = [Tracklet_w_reid(tlwh, s, cate, motion=self.motion, feat=feat) for
                            (tlwh, s, cate, feat) in zip(dets, scores_keep, cates, features_keep)]
            else:
                detections = [Tracklet(tlwh, s, cate, motion=self.motion) for
                            (tlwh, s, cate) in zip(dets, scores_keep, cates)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_tracklets'''
        # 更新跟踪状态 将当前跟踪的目标分为已确认（激活）和未确认（未激活）两类。
        unconfirmed = []
        tracked_tracklets = []  # type: list[Tracklet]
        for track in self.tracked_tracklets:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_tracklets.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        # 第一步：高分检测框的关联 创建跟踪目标池，结合当前跟踪目标和丢失目标，使用卡尔曼滤波预测每个跟踪目标的位置。
        tracklet_pool = joint_tracklets(tracked_tracklets, self.lost_tracklets)

        # Associate with high score detection boxes 2024年11月13日
        num_iteration = 2
        init_expand_scale = 0.1 # 将0.7改成0.25 0.10试试
        expand_scale_step = 0.1

        # Predict the current location with Kalman
        for tracklet in tracklet_pool:
            tracklet.predict()

        # Camera motion compensation
        # 进行相机运动补偿 应用相机运动补偿，以提高跟踪精度。
        for iteration in range(num_iteration):
            # 2024年11月13日
            cur_expand_scale = init_expand_scale + expand_scale_step*iteration

            warp = self.gmc.apply(ori_img, dets)
            self.gmc.multi_gmc(tracklet_pool, warp)
            self.gmc.multi_gmc(unconfirmed, warp)
            # 计算 IOU 距离 计算跟踪目标和检测目标之间的 IOU（Intersection over Union）距离，用于后续的匹配。
            ious_dists = eiou_distance(tracklet_pool, detections, cur_expand_scale) # 得到一个矩阵  2024年11月13日
            #print("tracklet_pool:", tracklet_pool, "detections:", detections)
            # print("ious_dists:", ious_dists)
            ious_dists_mask = (ious_dists > 0.5)  # high conf iou
            # 计算嵌入距离 如果启用了再识别，计算嵌入距离，并根据 IOU 和嵌入距离更新距离矩阵。
            if self.with_reid:
                # mixed cost matrix
                emb_dists = embedding_distance(tracklet_pool, detections) / 2.0
                raw_emb_dists = emb_dists.copy()
                emb_dists[emb_dists > 0.25] = 1.0
                emb_dists[ious_dists_mask] = 1.0
                dists = np.minimum(ious_dists, emb_dists)

            else:
                dists = ious_dists
        
            # 匹配跟踪目标和检测目标 使用匈牙利算法进行匹配。
            matches, u_track, u_detection = linear_assignment(dists, thresh=0.9)

            # 更新跟踪目标状态 更新匹配到的跟踪目标，如果状态是“跟踪中”，则更新；否则重新激活。
            for itracked, idet in matches:
                track = tracklet_pool[itracked]
                det = detections[idet]
                if track.state == TrackState.Tracked:
                    track.update(detections[idet], self.frame_id)
                    activated_tracklets.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_tracklets.append(track)


        '''
        warp = self.gmc.apply(ori_img, dets)
        self.gmc.multi_gmc(tracklet_pool, warp)
        self.gmc.multi_gmc(unconfirmed, warp)
        # 计算 IOU 距离 计算跟踪目标和检测目标之间的 IOU（Intersection over Union）距离，用于后续的匹配。
        ious_dists = iou_distance(tracklet_pool, detections) # 得到一个矩阵
        #print("tracklet_pool:", tracklet_pool, "detections:", detections)
        # print("ious_dists:", ious_dists)
        ious_dists_mask = (ious_dists > 0.5)  # high conf iou
        # 计算嵌入距离 如果启用了再识别，计算嵌入距离，并根据 IOU 和嵌入距离更新距离矩阵。
        if self.with_reid:
            # mixed cost matrix
            emb_dists = embedding_distance(tracklet_pool, detections) / 2.0
            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > 0.25] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)

        else:
            dists = ious_dists
        
        # 匹配跟踪目标和检测目标 使用匈牙利算法进行匹配。
        matches, u_track, u_detection = linear_assignment(dists, thresh=0.9)

        # 更新跟踪目标状态 更新匹配到的跟踪目标，如果状态是“跟踪中”，则更新；否则重新激活。
        for itracked, idet in matches:
            track = tracklet_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_tracklets.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_tracklets.append(track)
        '''

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        # 第二步：低分检测框的关联
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [Tracklet(tlwh, s, cate, motion=self.motion) for
                          (tlwh, s, cate) in zip(dets_second, scores_second, cates_second)]
        else:
            detections_second = []

        # 更新跟踪目标状态 对已跟踪目标和低分检测目标进行匹配。
        r_tracked_tracklets = [tracklet_pool[i] for i in u_track if tracklet_pool[i].state == TrackState.Tracked]
        dists = iou_distance(r_tracked_tracklets, detections_second)
        matches, u_track, u_detection_second = linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_tracklets[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_tracklets.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_tracklets.append(track)

        # 处理未确认跟踪目标 将未确认的目标标记为丢失。
        for it in u_track:
            track = r_tracked_tracklets[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_tracklets.append(track)


        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        # 处理未确认的跟踪目标 计算未确认跟踪目标与当前检测目标之间的 IOU 距离。
        detections = [detections[i] for i in u_detection]
        ious_dists = iou_distance(unconfirmed, detections)
        ious_dists_mask = (ious_dists > 0.5)

        if self.with_reid:
            emb_dists = embedding_distance(unconfirmed, detections) / 2.0
            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > 0.25] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

       
        # 匹配未确认跟踪目标 对未确认的跟踪目标进行匹配，更新状态。
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_tracklets.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_tracklets.append(track)

        """ Step 4: Init new tracklets"""
        # 初始化新的跟踪目标 对于新检测到的目标，根据阈值初始化新的跟踪目标。
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.frame_id)
            activated_tracklets.append(track)

        """ Step 5: Update state"""
        # 更新丢失状态 标记长时间丢失的目标为已删除。
        for track in self.lost_tracklets:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_tracklets.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        # 更新跟踪器的状态 更新跟踪器的状态，合并和清理跟踪目标列表。
        self.tracked_tracklets = [t for t in self.tracked_tracklets if t.state == TrackState.Tracked]
        self.tracked_tracklets = joint_tracklets(self.tracked_tracklets, activated_tracklets)
        self.tracked_tracklets = joint_tracklets(self.tracked_tracklets, refind_tracklets)
        self.lost_tracklets = sub_tracklets(self.lost_tracklets, self.tracked_tracklets)
        self.lost_tracklets.extend(lost_tracklets)
        self.lost_tracklets = sub_tracklets(self.lost_tracklets, self.removed_tracklets)
        self.removed_tracklets.extend(removed_tracklets)
        self.tracked_tracklets, self.lost_tracklets = remove_duplicate_tracklets(self.tracked_tracklets, self.lost_tracklets)
        # get scores of lost tracks
        # 返回当前活动的跟踪目标
        output_tracklets = [track for track in self.tracked_tracklets if track.is_activated]

        return output_tracklets

# 将两个跟踪目标列表合并，确保合并后的列表中没有重复的跟踪目标。
def joint_tracklets(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

# 从第一个跟踪目标列表中删除第二个列表中存在的跟踪目标。
def sub_tracklets(tlista, tlistb):
    tracklets = {}
    for t in tlista:
        tracklets[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if tracklets.get(tid, 0):
            del tracklets[tid]
    return list(tracklets.values())


# 该函数用于移除两个跟踪目标列表中重复的目标，基于它们之间的 IOU（Intersection Over Union）距离。
def remove_duplicate_tracklets(trackletsa, trackletsb):
    pdist = iou_distance(trackletsa, trackletsb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = trackletsa[p].frame_id - trackletsa[p].start_frame
        timeq = trackletsb[q].frame_id - trackletsb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(trackletsa) if not i in dupa]
    resb = [t for i, t in enumerate(trackletsb) if not i in dupb]
    return resa, resb