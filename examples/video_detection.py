"""
视频检测示例 - 完整功能版
使用GSE检测器处理视频文件或摄像头
支持高分辨率、ByteTrack跟踪、畸变矫正等功能

使用方法:
    # 基础使用（默认2560x1440@60fps, 带跟踪）
    python video_detection.py 0 --enable-tracking --device cuda
    
    # 自定义分辨率
    python video_detection.py 0 --width 1920 --height 1080 --fps 60
    
    # 启用畸变矫正
    python video_detection.py 0 --enable-tracking --undistort
"""

import cv2
import sys
import time
import argparse
import yaml
import numpy as np
from pathlib import Path
from collections import deque
from typing import Dict, Tuple, Optional

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from detector import GSEDetector
from byte_tracker import BYTETracker
from camera_io import CameraConfig, open_camera, get_actual_capture_params


def parse_args():
    """解析命令行参数"""
    p = argparse.ArgumentParser(description="GSE检测 - 完整功能版")
    
    # 视频源
    p.add_argument("source", nargs="?", default="0",
                   help="视频源：摄像头ID(如0)或视频文件路径")
    
    # 模型参数
    p.add_argument("--device", type=str, default="cuda",
                   choices=["cuda", "cpu"],
                   help="推理设备")
    p.add_argument("--conf", type=float, default=0.25,
                   help="置信度阈值")
    p.add_argument("--model-size", type=int, default=1280,
                   help="YOLO推理尺寸")
    
    # 摄像头参数（仅当source为摄像头ID时有效）
    p.add_argument("--width", type=int, default=2560,
                   help="相机宽度")
    p.add_argument("--height", type=int, default=1440,
                   help="相机高度")
    p.add_argument("--fps", type=int, default=60,
                   help="相机帧率")
    p.add_argument("--backend", type=str, default="dshow",
                   choices=["dshow", "msmf", "any"],
                   help="OpenCV后端（Windows推荐dshow）")
    p.add_argument("--fourcc", type=str, default="MJPG",
                   help="相机像素格式")
    p.add_argument("--buffersize", type=int, default=1,
                   help="采集缓冲区大小")
    
    # 跟踪参数
    p.add_argument("--enable-tracking", action="store_true",
                   help="启用ByteTrack多目标跟踪")
    p.add_argument("--track-thresh", type=float, default=0.5,
                   help="高置信度检测阈值")
    p.add_argument("--track-buffer", type=int, default=30,
                   help="轨迹缓冲帧数")
    p.add_argument("--match-thresh", type=float, default=0.8,
                   help="匹配IoU阈值")
    
    # 畸变矫正
    p.add_argument("--undistort", action="store_true",
                   help="启用畸变矫正")
    p.add_argument("--calib-config", type=str,
                   default="../tools/config/camera_calibration.yaml",
                   help="标定文件路径")
    
    return p.parse_args()


def load_camera_calibration(config_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """加载相机标定参数"""
    config_path = Path(__file__).parent.parent.parent / "tools" / "config" / "camera_calibration.yaml"
    
    if not config_path.exists():
        print(f"⚠️  标定文件不存在: {config_path}")
        return None, None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        camera_matrix = np.array(data['camera_matrix'])
        dist_coeffs = np.array(data['distortion_coefficients'])
        
        print(f"✅ 已加载标定参数")
        return camera_matrix, dist_coeffs
    except Exception as e:
        print(f"❌ 加载标定参数失败: {e}")
        return None, None


def draw_tracking_results(frame: np.ndarray, tracks: list, class_names: dict, conf_thresh: float = 0.25):
    """绘制跟踪结果"""
    for track in tracks:
        x1, y1, x2, y2 = map(int, track.tlbr)
        track_id = track.track_id
        score = track.score
        cls = int(track.cls)
        
        if score < conf_thresh:
            continue
        
        class_name = class_names.get(cls, f"class_{cls}")
        
        # 获取颜色
        from config import CLASS_COLORS
        color = CLASS_COLORS.get(cls, (0, 255, 255))
        
        # 绘制边框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签
        label = f"ID:{track_id} {class_name} {score:.2f}"
        
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        bg_x1, bg_y1 = x1, max(0, y1 - text_size[1] - 10)
        bg_x2, bg_y2 = x1 + text_size[0] + 10, y1
        
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main():
    args = parse_args()
    
    print("=" * 70)
    print("🚀 GSE检测系统 - 完整功能版")
    print("=" * 70)
    
    # 判断是摄像头还是视频文件
    try:
        camera_id = int(args.source)
        is_camera = True
        print(f"📷 使用摄像头 {camera_id}")
    except ValueError:
        is_camera = False
        print(f"📹 使用视频文件: {args.source}")
    
    # 打开视频源
    if is_camera:
        cap_cfg = CameraConfig(
            camera_id=camera_id,
            width=args.width,
            height=args.height,
            fps=args.fps,
            backend=args.backend,
            fourcc=args.fourcc,
            buffersize=args.buffersize
        )
        cap = open_camera(cap_cfg)
        if cap.isOpened():
            actual = get_actual_capture_params(cap)
            print(f"✅ 相机已打开: {actual['width']}x{actual['height']} @ {actual['fps']:.1f}FPS (FOURCC={args.fourcc})")
    else:
        cap = cv2.VideoCapture(args.source)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"✅ 视频已打开: {width}x{height} @ {fps:.1f}FPS")
    
    if not cap.isOpened():
        print(f"❌ 无法打开视频源")
        return
    
    # 初始化检测器
    print("\n初始化检测器...")
    detector = GSEDetector(
        device=args.device,
        conf_threshold=args.conf,
        input_size=args.model_size
    )
    
    # 初始化ByteTrack跟踪器
    tracker = None
    if args.enable_tracking:
        tracker = BYTETracker(
            track_thresh=args.track_thresh,
            track_buffer=args.track_buffer,
            match_thresh=args.match_thresh,
            frame_rate=args.fps
        )
        print(f"✅ ByteTrack已初始化 (thresh={args.track_thresh}, buffer={args.track_buffer})")
    
    # 加载畸变矫正参数
    camera_matrix, dist_coeffs = None, None
    undistort_enabled = args.undistort
    if undistort_enabled:
        camera_matrix, dist_coeffs = load_camera_calibration(args.calib_config)
    
    # FPS计算
    fps_queue = deque(maxlen=30)
    frame_count = 0
    map1, map2 = None, None
    
    print("\n按键说明:")
    print("  Q - 退出")
    print("  S - 保存当前帧")
    print("  U - 切换畸变矫正")
    if tracker:
        print("  T - 切换跟踪")
    print("-" * 70)
    
    tracking_enabled = args.enable_tracking
    
    try:
        while True:
            start_time = time.time()
            
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print("视频结束或读取失败")
                break
            
            frame_count += 1
            h, w = frame.shape[:2]
            
            # 畸变矫正
            if undistort_enabled and camera_matrix is not None:
                if map1 is None or map2 is None:
                    new_camera_matrix, _roi = cv2.getOptimalNewCameraMatrix(
                        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
                    )
                    map1, map2 = cv2.initUndistortRectifyMap(
                        camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_16SC2
                    )
                frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
            
            # 执行检测
            if tracking_enabled and tracker:
                # 使用跟踪模式
                raw_results = detector.detect(frame, return_raw=True)
                
                # 转换为ByteTrack格式 [x1, y1, x2, y2, score, class]
                det_list = []
                if len(raw_results) > 0 and raw_results[0].boxes is not None:
                    boxes = raw_results[0].boxes
                    if len(boxes) > 0:
                        xyxy = boxes.xyxy.cpu().numpy()
                        confs = boxes.conf.cpu().numpy()
                        clss = boxes.cls.cpu().numpy()
                        det_list = np.concatenate([xyxy, confs[:, None], clss[:, None]], axis=1)
                
                detections_np = np.array(det_list) if len(det_list) > 0 else np.empty((0, 6))
                tracks = tracker.update(detections_np)
                
                # 绘制跟踪结果
                result_frame = frame.copy()
                draw_tracking_results(result_frame, tracks, detector.class_names, args.conf)
                num_objects = len(tracks)
            else:
                # 仅检测模式
                detections = detector.detect(frame)
                result_frame = detector.draw_results(frame, detections)
                num_objects = len(detections)
            
            # 计算FPS
            elapsed = time.time() - start_time
            fps_queue.append(1.0 / elapsed if elapsed > 0 else 0)
            current_fps = sum(fps_queue) / len(fps_queue)
            
            # 显示信息
            info_lines = [
                f"FPS: {current_fps:.1f}",
                f"Objects: {num_objects}",
                f"Resolution: {w}x{h}",
                f"Undistort: {'ON' if undistort_enabled else 'OFF'}",
                f"Tracking: {'ON' if tracking_enabled else 'OFF'}",
                f"Device: {args.device.upper()}"
            ]
            
            y_offset = 30
            for line in info_lines:
                cv2.putText(result_frame, line, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30
            
            # 自动缩放显示（如果分辨率超过1920x1080）
            display_frame = result_frame
            if w > 1920 or h > 1080:
                scale = min(1920/w, 1080/h)
                display_w = int(w * scale)
                display_h = int(h * scale)
                display_frame = cv2.resize(result_frame, (display_w, display_h))
            
            # 显示结果
            cv2.imshow("GSE Detection (Q:Quit, S:Save, U:Undistort, T:Tracking)", display_frame)
            
            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\n退出...")
                break
            elif key == ord('s') or key == ord('S'):
                filename = f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(filename, result_frame)
                print(f"💾 已保存: {filename}")
            elif key == ord('u') or key == ord('U'):
                if camera_matrix is not None:
                    undistort_enabled = not undistort_enabled
                    map1, map2 = None, None  # 重置映射
                    print(f"🔄 畸变矫正: {'启用' if undistort_enabled else '禁用'}")
                else:
                    print("⚠️  未加载标定参数")
            elif key == ord('t') or key == ord('T'):
                if tracker:
                    tracking_enabled = not tracking_enabled
                    if not tracking_enabled:
                        tracker.reset()
                    print(f"🎯 跟踪: {'启用' if tracking_enabled else '禁用'}")
    
    except KeyboardInterrupt:
        print("\n🛑 程序中断")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 70)
        print("📊 统计信息:")
        print(f"  总帧数: {frame_count}")
        print(f"  平均FPS: {sum(fps_queue)/len(fps_queue):.2f}" if fps_queue else "  平均FPS: N/A")
        if tracker:
            print(f"  总轨迹数: {tracker.track_id_count}")
        print("=" * 70)
        print("✅ 程序结束")


if __name__ == "__main__":
    main()
