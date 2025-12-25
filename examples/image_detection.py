"""
图像检测示例
使用GSE检测器处理单张图像
"""

import cv2
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from detector import GSEDetector


def main():
    # 图像路径
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("请提供图像路径")
        print("用法: python image_detection.py <image_path>")
        return
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图像: {image_path}")
        return
    
    print(f"图像尺寸: {image.shape[1]}x{image.shape[0]}")
    
    # 初始化检测器
    print("\n初始化检测器...")
    detector = GSEDetector(
        device="cuda",  # 使用GPU，如无GPU改为"cpu"
        conf_threshold=0.25
    )
    
    # 执行检测
    print("执行检测...")
    detections = detector.detect(image)
    
    # 打印结果
    print(f"\n✅ 检测到 {len(detections)} 个目标:")
    for i, det in enumerate(detections, 1):
        print(f"  {i}. {det['class_name']} (置信度: {det['confidence']:.2f})")
    
    # 统计信息
    stats = detector.get_statistics(detections)
    print("\n📊 类别统计:")
    for class_name, count in stats.items():
        print(f"  {class_name}: {count}")
    
    # 绘制结果
    result_img = detector.draw_results(image, detections)
    
    # 保存结果
    output_path = "detection_result.jpg"
    cv2.imwrite(output_path, result_img)
    print(f"\n💾 结果已保存到: {output_path}")
    
    # 显示结果
    cv2.imshow("Detection Result (Press Q to exit)", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
