"""ByteTrack: Multi-Object Tracking by Associating Every Detection Box
基于 FoundationVision/ByteTrack 的简化实现
原始论文: https://arxiv.org/abs/2110.06864
"""

import numpy as np
from collections import deque
from typing import List, Tuple, Optional


class KalmanFilter:
    """简化的卡尔曼滤波器用于边界框跟踪 (x, y, w, h, vx, vy, vw, vh)"""
    
    def __init__(self):
        # 状态维度: [x, y, w, h, vx, vy, vw, vh]
        self.ndim = 8
        self.dt = 1.0
        
        # 状态转移矩阵 (运动模型)
        self.F = np.eye(self.ndim)
        for i in range(4):
            self.F[i, i+4] = self.dt
        
        # 测量矩阵 (只能观测位置和尺寸)
        self.H = np.eye(4, self.ndim)
        
        # 过程噪声协方差
        self.Q = np.eye(self.ndim)
        self.Q[4:, 4:] *= 0.01  # 速度噪声
        
        # 测量噪声协方差
        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0  # 尺寸测量不太准确
        
    def initiate(self, measurement: np.ndarray):
        """初始化跟踪器状态
        
        Args:
            measurement: [x, y, w, h] 初始检测框
        
        Returns:
            mean: 状态均值
            covariance: 状态协方差
        """
        mean = np.zeros(self.ndim)
        mean[:4] = measurement
        
        covariance = np.eye(self.ndim)
        covariance[4:, 4:] *= 1000.0  # 速度初始不确定性大
        covariance[:4, :4] *= 10.0
        
        return mean, covariance
    
    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        """预测下一帧状态"""
        mean = self.F @ mean
        covariance = self.F @ covariance @ self.F.T + self.Q
        return mean, covariance
    
    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray):
        """更新状态基于新观测"""
        # 投影到测量空间
        projected_mean = self.H @ mean
        projected_cov = self.H @ covariance @ self.H.T
        
        # 残差
        innovation = measurement - projected_mean
        
        # 卡尔曼增益
        S = projected_cov + self.R
        K = covariance @ self.H.T @ np.linalg.inv(S)
        
        # 更新
        mean = mean + K @ innovation
        covariance = covariance - K @ self.H @ covariance
        
        return mean, covariance


def bbox_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """计算两个边界框的IoU
    
    Args:
        bbox1, bbox2: [x1, y1, x2, y2]
    
    Returns:
        IoU值
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    union_area = bbox1_area + bbox2_area - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area


def iou_distance(tracks: List['STrack'], detections: List['STrack']) -> np.ndarray:
    """计算tracks和detections之间的IoU距离矩阵"""
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)))
    
    cost_matrix = np.zeros((len(tracks), len(detections)))
    
    for i, track in enumerate(tracks):
        for j, det in enumerate(detections):
            cost_matrix[i, j] = 1 - bbox_iou(track.tlbr, det.tlbr)
    
    return cost_matrix


def linear_assignment(cost_matrix: np.ndarray, thresh: float) -> Tuple[np.ndarray, List[int], List[int]]:
    """使用贪心算法进行线性分配 (简化版,不需要lap包)
    
    Args:
        cost_matrix: 代价矩阵 [N, M]
        thresh: 匹配阈值
    
    Returns:
        matches: 匹配对 [[track_idx, det_idx], ...]
        unmatched_a: 未匹配的track索引
        unmatched_b: 未匹配的detection索引
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
    
    matches = []
    
    # 复制矩阵用于修改
    cost = cost_matrix.copy()
    
    # 贪心匹配
    while True:
        # 找到最小代价
        min_val = cost.min()
        if min_val > thresh:
            break
        
        min_idx = np.unravel_index(cost.argmin(), cost.shape)
        matches.append([min_idx[0], min_idx[1]])
        
        # 标记已匹配
        cost[min_idx[0], :] = thresh + 1
        cost[:, min_idx[1]] = thresh + 1
    
    matches = np.array(matches) if matches else np.empty((0, 2), dtype=int)
    
    # 未匹配的索引
    matched_a = set(matches[:, 0]) if len(matches) > 0 else set()
    matched_b = set(matches[:, 1]) if len(matches) > 0 else set()
    
    unmatched_a = [i for i in range(cost_matrix.shape[0]) if i not in matched_a]
    unmatched_b = [j for j in range(cost_matrix.shape[1]) if j not in matched_b]
    
    return matches, unmatched_a, unmatched_b


class TrackState:
    """轨迹状态枚举"""
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class STrack:
    """单目标轨迹"""
    
    shared_kalman = KalmanFilter()
    
    def __init__(self, tlbr: np.ndarray, score: float, cls: int):
        """
        Args:
            tlbr: [x1, y1, x2, y2]
            score: 置信度
            cls: 类别ID
        """
        # 转换为 [cx, cy, w, h]
        self._tlbr = tlbr.copy()
        w = tlbr[2] - tlbr[0]
        h = tlbr[3] - tlbr[1]
        cx = tlbr[0] + w / 2
        cy = tlbr[1] + h / 2
        
        # 卡尔曼滤波器状态
        self.mean, self.covariance = self.shared_kalman.initiate(np.array([cx, cy, w, h]))
        
        self.score = score
        self.cls = cls
        self.tracklet_len = 0
        self.state = TrackState.New
        self.is_activated = False
        self.track_id = 0
        self.frame_id = 0
        self.start_frame = 0
    
    @property
    def tlbr(self) -> np.ndarray:
        """获取 [x1, y1, x2, y2] 格式的边界框"""
        cx, cy, w, h = self.mean[:4]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return np.array([x1, y1, x2, y2])
    
    def predict(self):
        """卡尔曼滤波预测"""
        self.mean, self.covariance = self.shared_kalman.predict(self.mean, self.covariance)
    
    def update(self, new_track: 'STrack', frame_id: int):
        """更新轨迹"""
        self.frame_id = frame_id
        self.tracklet_len += 1
        
        # 提取新的测量值
        tlbr = new_track._tlbr
        w = tlbr[2] - tlbr[0]
        h = tlbr[3] - tlbr[1]
        cx = tlbr[0] + w / 2
        cy = tlbr[1] + h / 2
        
        # 卡尔曼更新
        measurement = np.array([cx, cy, w, h])
        self.mean, self.covariance = self.shared_kalman.update(self.mean, self.covariance, measurement)
        
        self.score = new_track.score
        self.cls = new_track.cls
        self.state = TrackState.Tracked
        self.is_activated = True
    
    def activate(self, frame_id: int, track_id: int):
        """激活新轨迹"""
        self.track_id = track_id
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
    
    def re_activate(self, new_track: 'STrack', frame_id: int, new_id: bool = False):
        """重新激活丢失的轨迹"""
        # 提取新的测量值
        tlbr = new_track._tlbr
        w = tlbr[2] - tlbr[0]
        h = tlbr[3] - tlbr[1]
        cx = tlbr[0] + w / 2
        cy = tlbr[1] + h / 2
        
        # 卡尔曼更新
        measurement = np.array([cx, cy, w, h])
        self.mean, self.covariance = self.shared_kalman.update(self.mean, self.covariance, measurement)
        
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.score = new_track.score
        self.cls = new_track.cls
        
        if new_id:
            self.track_id = new_track.track_id
    
    def mark_lost(self):
        """标记为丢失"""
        self.state = TrackState.Lost
    
    def mark_removed(self):
        """标记为移除"""
        self.state = TrackState.Removed


class BYTETracker:
    """ByteTrack多目标跟踪器"""
    
    def __init__(self, 
                 track_thresh: float = 0.5,
                 track_buffer: int = 30,
                 match_thresh: float = 0.8,
                 frame_rate: int = 30):
        """
        Args:
            track_thresh: 高置信度检测阈值
            track_buffer: 轨迹保留的最大帧数
            match_thresh: 匹配IoU阈值
            frame_rate: 视频帧率
        """
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)
        
        self.frame_id = 0
        self.tracked_stracks: List[STrack] = []  # 正在跟踪的轨迹
        self.lost_stracks: List[STrack] = []     # 丢失的轨迹
        self.removed_stracks: List[STrack] = []  # 移除的轨迹
        
        self.track_id_count = 0
    
    def update(self, detections: np.ndarray) -> List[STrack]:
        """更新跟踪器
        
        Args:
            detections: [N, 6] 格式 [x1, y1, x2, y2, score, class]
        
        Returns:
            当前帧的活跃轨迹列表
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        if len(detections) == 0:
            # 没有检测：预测轨迹并将当前 tracked 标记为丢失，但保留历史 lost 轨迹，避免“空帧即全量重置”。
            for track in self.tracked_stracks:
                track.predict()
            for track in self.lost_stracks:
                track.predict()

            for track in self.tracked_stracks:
                track.mark_lost()
                lost_stracks.append(track)

            self.tracked_stracks = []
            self.lost_stracks.extend(lost_stracks)

            # 移除长时间丢失的轨迹
            for track in self.lost_stracks:
                if self.frame_id - track.frame_id > self.max_time_lost:
                    track.mark_removed()
                    removed_stracks.append(track)

            self.lost_stracks = [t for t in self.lost_stracks if t.state == TrackState.Lost]
            self.removed_stracks.extend(removed_stracks)
            return []
        
        # 分离高低置信度检测
        scores = detections[:, 4]
        remain_inds = scores >= self.track_thresh
        dets = detections[remain_inds]
        scores_keep = scores[remain_inds]
        
        remain_inds_low = np.logical_and(scores >= 0.1, scores < self.track_thresh)
        dets_low = detections[remain_inds_low]
        scores_low = scores[remain_inds_low]
        
        # 创建检测对象
        detections_high = [STrack(det[:4], det[4], int(det[5])) for det in dets]
        detections_low = [STrack(det[:4], det[4], int(det[5])) for det in dets_low]
        
        # 1. 预测所有轨迹
        for track in self.tracked_stracks:
            track.predict()
        for track in self.lost_stracks:
            track.predict()
        
        # 2. 第一次关联: tracked tracks <-> high score detections
        cost_matrix = iou_distance(self.tracked_stracks, detections_high)
        matches, u_track, u_detection = linear_assignment(cost_matrix, thresh=self.match_thresh)
        
        # 更新匹配的轨迹
        for itracked, idet in matches:
            track = self.tracked_stracks[itracked]
            det = detections_high[idet]
            track.update(det, self.frame_id)
            activated_stracks.append(track)
        
        # 3. 第二次关联: remaining tracked tracks <-> low score detections
        r_tracked_stracks = [self.tracked_stracks[i] for i in u_track]
        
        cost_matrix_low = iou_distance(r_tracked_stracks, detections_low)
        matches_low, u_track_low, u_detection_low = linear_assignment(cost_matrix_low, thresh=0.5)
        
        for itracked, idet in matches_low:
            track = r_tracked_stracks[itracked]
            det = detections_low[idet]
            track.update(det, self.frame_id)
            activated_stracks.append(track)
        
        # 标记未匹配的tracked轨迹为丢失
        for it in u_track_low:
            track = r_tracked_stracks[it]
            track.mark_lost()
            lost_stracks.append(track)
        
        # 4. 第三次关联: lost tracks <-> high score remaining detections
        detections_remain = [detections_high[i] for i in u_detection]
        
        cost_matrix_lost = iou_distance(self.lost_stracks, detections_remain)
        matches_lost, u_lost, u_detection_remain = linear_assignment(cost_matrix_lost, thresh=self.match_thresh)
        
        for ilost, idet in matches_lost:
            track = self.lost_stracks[ilost]
            det = detections_remain[idet]
            track.re_activate(det, self.frame_id, new_id=False)
            refind_stracks.append(track)
        
        # 5. 初始化新轨迹
        for inew in u_detection_remain:
            track = detections_remain[inew]
            if track.score >= self.track_thresh:
                self.track_id_count += 1
                track.activate(self.frame_id, self.track_id_count)
                activated_stracks.append(track)
        
        # 6. 移除长时间丢失的轨迹
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
        
        # 更新轨迹池
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = activated_stracks + refind_stracks
        self.lost_stracks = [t for t in self.lost_stracks if t.state == TrackState.Lost]
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = [t for t in self.lost_stracks if t not in removed_stracks]
        self.removed_stracks.extend(removed_stracks)
        
        # 返回当前活跃轨迹
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        
        return output_stracks

    def predict_only(self) -> List[STrack]:
        """只做卡尔曼预测，不进行检测关联。

        用于推理降频/跳帧时保持画面流畅（不建议长期无检测）。
        """
        self.frame_id += 1
        for track in self.tracked_stracks:
            track.predict()
        for track in self.lost_stracks:
            track.predict()
        return [track for track in self.tracked_stracks if track.is_activated]
    
    def reset(self):
        """重置跟踪器"""
        self.frame_id = 0
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.track_id_count = 0
