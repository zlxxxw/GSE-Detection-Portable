"""
API集成示例
展示如何将GSE检测器集成到其他项目中
"""

import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from detector import GSEDetector
import cv2


def example_basic_detection():
    """基础检测示例"""
    print("=" * 60)
    print("示例1: 基础检测")
    print("=" * 60)
    
    # 初始化检测器
    detector = GSEDetector()
    
    # 读取图像
    image = cv2.imread("test.jpg")
    
    # 执行检测
    detections = detector.detect(image)
    
    # 处理结果
    for det in detections:
        print(f"检测到 {det['class_name']}, 置信度: {det['confidence']:.2f}")
    
    # 绘制结果
    result = detector.draw_results(image, detections)
    cv2.imwrite("result.jpg", result)


def example_class_filtering():
    """类别过滤示例"""
    print("\n" + "=" * 60)
    print("示例2: 类别过滤")
    print("=" * 60)
    
    detector = GSEDetector()
    image = cv2.imread("test.jpg")
    
    # 检测所有目标
    all_detections = detector.detect(image)
    print(f"总检测数: {len(all_detections)}")
    
    # 只保留飞机
    airplanes = detector.filter_by_class(
        all_detections,
        class_names=["airplane"]
    )
    print(f"飞机数量: {len(airplanes)}")
    
    # 只保留地面设备和人员
    ground_objects = detector.filter_by_class(
        all_detections,
        class_names=["Ground_Crew", "Unmaned_GSE", "Galley_Truck"]
    )
    print(f"地面目标数量: {len(ground_objects)}")


def example_custom_visualization():
    """自定义可视化示例"""
    print("\n" + "=" * 60)
    print("示例3: 自定义可视化")
    print("=" * 60)
    
    detector = GSEDetector()
    image = cv2.imread("test.jpg")
    
    # 检测
    detections = detector.detect(image, conf_threshold=0.5)  # 使用更高的阈值
    
    # 自定义绘制
    result = detector.draw_results(
        image,
        detections,
        thickness=3,           # 更粗的线条
        font_scale=0.8,        # 更大的字体
        show_conf=True         # 显示置信度
    )
    
    cv2.imwrite("custom_result.jpg", result)


def example_batch_processing():
    """批量处理示例"""
    print("\n" + "=" * 60)
    print("示例4: 批量处理")
    print("=" * 60)
    
    detector = GSEDetector()
    
    # 图像列表
    image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
    
    all_stats = {}
    for img_path in image_paths:
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        # 检测
        detections = detector.detect(image)
        stats = detector.get_statistics(detections)
        
        print(f"\n{img_path}:")
        for class_name, count in stats.items():
            print(f"  {class_name}: {count}")
            all_stats[class_name] = all_stats.get(class_name, 0) + count
    
    print("\n总计:")
    for class_name, count in all_stats.items():
        print(f"  {class_name}: {count}")


def example_api_integration():
    """API集成示例 - 返回JSON格式"""
    print("\n" + "=" * 60)
    print("示例5: API集成 (JSON输出)")
    print("=" * 60)
    
    import json
    
    detector = GSEDetector()
    image = cv2.imread("test.jpg")
    
    # 检测
    detections = detector.detect(image)
    
    # 转换为API响应格式
    api_response = {
        "status": "success",
        "count": len(detections),
        "detections": detections,
        "statistics": detector.get_statistics(detections)
    }
    
    # 输出JSON
    print(json.dumps(api_response, indent=2, ensure_ascii=False))


def example_realtime_callback():
    """实时处理回调示例"""
    print("\n" + "=" * 60)
    print("示例6: 实时回调处理")
    print("=" * 60)
    
    def on_detection(detections, frame_id):
        """检测回调函数"""
        print(f"帧 {frame_id}: 检测到 {len(detections)} 个目标")
        for det in detections:
            if det['confidence'] > 0.8:  # 高置信度报警
                print(f"  ⚠️ 高置信度: {det['class_name']} ({det['confidence']:.2f})")
    
    detector = GSEDetector()
    
    # 模拟视频流处理
    cap = cv2.VideoCapture(0)
    frame_id = 0
    
    while frame_id < 100:  # 处理100帧
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_id += 1
        detections = detector.detect(frame)
        on_detection(detections, frame_id)
    
    cap.release()


if __name__ == "__main__":
    print("\n🚀 GSE检测器 - API集成示例\n")
    
    # 运行各个示例
    # example_basic_detection()
    # example_class_filtering()
    # example_custom_visualization()
    # example_batch_processing()
    example_api_integration()
    # example_realtime_callback()
    
    print("\n✅ 示例完成!")
