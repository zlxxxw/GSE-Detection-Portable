"""
精简YOLO检测器 - 可移植版本
Portable YOLO Detector for Airport GSE Detection

使用示例:
    from detector import GSEDetector
    
    detector = GSEDetector()
    results = detector.detect(image)
    annotated_image = detector.draw_results(image, results)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import config


class GSEDetector:
    """机场GSE检测器"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        input_size: Optional[int] = None
    ):
        """
        初始化检测器
        
        Args:
            model_path: 模型路径，默认使用config中的配置
            device: 设备 ("cuda", "cpu", "mps")，默认使用config中的配置
            conf_threshold: 置信度阈值，默认使用config中的配置
            input_size: 输入尺寸，默认使用config中的配置
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "需要安装 ultralytics: pip install ultralytics"
            )
        
        # 配置参数
        self.model_path = model_path or config.MODEL_PATH
        self.device = device or config.DEVICE
        self.conf_threshold = conf_threshold or config.CONFIDENCE_THRESHOLD
        self.input_size = input_size or config.INPUT_SIZE
        self.class_names = config.CLASS_NAMES
        self.class_colors = config.CLASS_COLORS
        
        # 加载模型
        model_full_path = Path(__file__).parent / self.model_path
        if not model_full_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_full_path}")
        
        print(f"正在加载模型: {model_full_path}")
        self.model = YOLO(str(model_full_path))
        self.model.to(self.device)
        print(f"✅ 模型已加载 (device: {self.device})")
    
    def detect(
        self,
        image: np.ndarray,
        conf_threshold: Optional[float] = None,
        return_raw: bool = False
    ) -> List[Dict]:
        """
        检测图像中的目标
        
        Args:
            image: 输入图像 (BGR格式)
            conf_threshold: 置信度阈值，None则使用默认值
            return_raw: 是否返回原始结果对象
        
        Returns:
            检测结果列表，每个结果包含:
                - bbox: [x1, y1, x2, y2] 边界框坐标
                - confidence: float 置信度
                - class_id: int 类别ID
                - class_name: str 类别名称
        """
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        
        # 执行推理
        results = self.model(
            image,
            imgsz=self.input_size,
            conf=conf,
            device=self.device,
            verbose=False
        )
        
        if return_raw:
            return results
        
        # 解析结果
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': self.class_names.get(class_id, f"class_{class_id}")
                })
        
        return detections
    
    def draw_results(
        self,
        image: np.ndarray,
        detections: List[Dict],
        thickness: int = 2,
        font_scale: float = 0.6,
        show_conf: bool = True
    ) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        Args:
            image: 输入图像
            detections: detect()返回的检测结果
            thickness: 线条粗细
            font_scale: 字体大小
            show_conf: 是否显示置信度
        
        Returns:
            标注后的图像
        """
        result_img = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            confidence = det['confidence']
            class_id = det['class_id']
            class_name = det['class_name']
            
            # 获取颜色
            color = self.class_colors.get(class_id, (0, 255, 255))
            
            # 绘制边界框
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, thickness)
            
            # 绘制标签
            if show_conf:
                label = f"{class_name} {confidence:.2f}"
            else:
                label = class_name
            
            # 计算标签背景框
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            
            # 绘制标签背景
            cv2.rectangle(
                result_img,
                (x1, y1 - label_h - baseline - 5),
                (x1 + label_w + 5, y1),
                color,
                -1
            )
            
            # 绘制标签文字
            cv2.putText(
                result_img,
                label,
                (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness
            )
        
        return result_img
    
    def filter_by_class(
        self,
        detections: List[Dict],
        class_ids: Optional[List[int]] = None,
        class_names: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        按类别过滤检测结果
        
        Args:
            detections: 检测结果
            class_ids: 保留的类别ID列表
            class_names: 保留的类别名称列表
        
        Returns:
            过滤后的检测结果
        """
        if class_ids is None and class_names is None:
            return detections
        
        filtered = []
        for det in detections:
            if class_ids and det['class_id'] in class_ids:
                filtered.append(det)
            elif class_names and det['class_name'] in class_names:
                filtered.append(det)
        
        return filtered
    
    def get_statistics(self, detections: List[Dict]) -> Dict[str, int]:
        """
        统计各类别的检测数量
        
        Args:
            detections: 检测结果
        
        Returns:
            类别统计字典 {class_name: count}
        """
        stats = {}
        for det in detections:
            class_name = det['class_name']
            stats[class_name] = stats.get(class_name, 0) + 1
        return stats


def main():
    """测试函数"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python detector.py <image_path>")
        print("示例: python detector.py test.jpg")
        return
    
    # 读取图像
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图像: {image_path}")
        return
    
    # 初始化检测器
    print("初始化检测器...")
    detector = GSEDetector()
    
    # 执行检测
    print("执行检测...")
    detections = detector.detect(image)
    
    # 打印结果
    print(f"\n检测到 {len(detections)} 个目标:")
    stats = detector.get_statistics(detections)
    for class_name, count in stats.items():
        print(f"  {class_name}: {count}")
    
    # 绘制结果
    result_img = detector.draw_results(image, detections)
    
    # 保存结果
    output_path = "detection_result.jpg"
    cv2.imwrite(output_path, result_img)
    print(f"\n✅ 结果已保存到: {output_path}")
    
    # 显示结果
    cv2.imshow("Detection Result", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
