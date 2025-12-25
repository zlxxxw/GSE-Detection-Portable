# 机场GSE检测模型 - 可移植版本

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**精简、可移植的机场地面设备 (GSE) 检测模型**

本包提供训练好的YOLOv11模型，用于检测机场场景中的目标，包括：
- 🚚 **Galley_Truck** (餐车)
- 🤖 **Unmaned_GSE** (无人地面设备)
- 👷 **Ground_Crew** (地勤人员)
- ✈️ **airplane** (飞机)

---

## 📦 文件结构

```
portable_model/
├── config.py              # 模型配置（类别、颜色、参数）
├── detector.py            # 核心检测器类
├── requirements.txt       # 依赖清单
├── README.md             # 本文档
├── weights/
│   └── gse_detection_v11.pt   # 训练好的模型权重
└── examples/
    ├── image_detection.py      # 图像检测示例
    ├── video_detection.py      # 视频检测示例
    └── api_integration.py      # API集成示例
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**GPU加速 (推荐):**
```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. 基础使用

```python
from detector import GSEDetector
import cv2

# 初始化检测器
detector = GSEDetector(device="cuda")  # 或 "cpu"

# 读取图像
image = cv2.imread("test.jpg")

# 执行检测
detections = detector.detect(image)

# 绘制结果
result = detector.draw_results(image, detections)

# 保存结果
cv2.imwrite("result.jpg", result)
```

### 3. 运行示例

**检测图像:**
```bash
python examples/image_detection.py your_image.jpg
```

**检测视频或摄像头:**
```bash
# 摄像头
python examples/video_detection.py 0

# 视频文件
python examples/video_detection.py video.mp4
```

---

## 📖 API文档

### GSEDetector类

#### 初始化
```python
detector = GSEDetector(
    model_path=None,        # 模型路径，默认使用config中的配置
    device="cuda",          # 设备: "cuda", "cpu", "mps"
    conf_threshold=0.25,    # 置信度阈值
    input_size=1280         # 输入尺寸
)
```

#### 核心方法

**1. detect() - 执行检测**
```python
detections = detector.detect(
    image,                  # 输入图像 (BGR格式)
    conf_threshold=None,    # 覆盖默认阈值
    return_raw=False        # 是否返回原始结果
)

# 返回格式:
# [
#     {
#         'bbox': [x1, y1, x2, y2],
#         'confidence': 0.95,
#         'class_id': 3,
#         'class_name': 'airplane'
#     },
#     ...
# ]
```

**2. draw_results() - 绘制结果**
```python
result_img = detector.draw_results(
    image,                  # 输入图像
    detections,             # detect()返回的结果
    thickness=2,            # 线条粗细
    font_scale=0.6,         # 字体大小
    show_conf=True          # 是否显示置信度
)
```

**3. filter_by_class() - 类别过滤**
```python
# 按类别ID过滤
airplanes = detector.filter_by_class(
    detections,
    class_ids=[3]  # 只保留飞机
)

# 按类别名称过滤
ground_staff = detector.filter_by_class(
    detections,
    class_names=["Ground_Crew"]
)
```

**4. get_statistics() - 统计信息**
```python
stats = detector.get_statistics(detections)
# 返回: {'airplane': 2, 'Ground_Crew': 5, ...}
```

---

## 🔧 配置文件 (config.py)

```python
# 修改类别名称
CLASS_NAMES = {
    0: "Galley_Truck",
    1: "Unmaned_GSE",
    2: "Ground_Crew",
    3: "airplane"
}

# 修改颜色 (BGR格式)
CLASS_COLORS = {
    0: (0, 0, 255),      # 红色
    1: (0, 0, 255),      # 红色
    2: (255, 0, 0),      # 蓝色
    3: (0, 255, 0)       # 绿色
}

# 修改默认参数
CONFIDENCE_THRESHOLD = 0.25
INPUT_SIZE = 1280
DEVICE = "cuda"
```

---

## 💡 使用场景

### 1. Web API集成
```python
from flask import Flask, request, jsonify
from detector import GSEDetector
import cv2
import numpy as np

app = Flask(__name__)
detector = GSEDetector()

@app.route('/detect', methods=['POST'])
def detect_api():
    # 接收图像
    file = request.files['image']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    
    # 检测
    detections = detector.detect(image)
    stats = detector.get_statistics(detections)
    
    return jsonify({
        'count': len(detections),
        'detections': detections,
        'statistics': stats
    })
```

### 2. 实时监控
```python
import cv2
from detector import GSEDetector

detector = GSEDetector()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    detections = detector.detect(frame)
    result = detector.draw_results(frame, detections)
    
    cv2.imshow('Monitor', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 3. 批量处理
```python
from pathlib import Path
from detector import GSEDetector
import cv2

detector = GSEDetector()
image_dir = Path("images")

for img_path in image_dir.glob("*.jpg"):
    image = cv2.imread(str(img_path))
    detections = detector.detect(image)
    
    # 保存结果
    result = detector.draw_results(image, detections)
    output_path = f"results/{img_path.name}"
    cv2.imwrite(output_path, result)
```

---

## ⚡ 性能优化

### GPU加速
- 确保安装CUDA版本的PyTorch
- 使用 `device="cuda"` 初始化检测器
- 推荐使用NVIDIA RTX系列GPU

### 推理速度
- RTX A4000 (16GB): ~100 FPS @ 1280x720
- RTX 3060 (12GB): ~70 FPS @ 1280x720
- CPU (i7-12700): ~15 FPS @ 1280x720

### 降低延迟
```python
# 使用较小的输入尺寸
detector = GSEDetector(input_size=640)  # 默认1280

# 提高置信度阈值
detections = detector.detect(image, conf_threshold=0.5)
```

---

## 📊 模型信息

- **架构**: YOLOv11s
- **输入尺寸**: 1280x1280
- **训练数据**: 171张机场场景图像
- **类别数**: 4
- **精度**: mAP@0.5 > 0.90

---

## 🔄 迁移到其他项目

### 方法1: 直接复制
```bash
# 复制整个portable_model文件夹到目标项目
cp -r portable_model /path/to/your/project/
```

### 方法2: 作为子模块使用
```python
# 在目标项目中
import sys
sys.path.insert(0, '/path/to/portable_model')

from detector import GSEDetector
```

### 方法3: 安装为包
```bash
# 在portable_model目录下
pip install -e .
```

---

## 🐛 故障排除

### 1. CUDA不可用
```python
# 检查CUDA
import torch
print(torch.cuda.is_available())

# 如无GPU，使用CPU
detector = GSEDetector(device="cpu")
```

### 2. 模型加载失败
```python
# 检查模型路径
from pathlib import Path
model_path = Path("weights/gse_detection_v11.pt")
print(f"模型存在: {model_path.exists()}")
```

### 3. OpenCV显示问题
```bash
# Windows可能需要额外依赖
pip install opencv-python-headless
```

---

## 📝 更新日志

**v1.0.0** (2025-12-25)
- ✅ 初始版本发布
- ✅ YOLOv11s模型
- ✅ 支持GPU加速
- ✅ 完整API和示例

---

## 📄 许可证

MIT License - 可自由用于商业和个人项目

---

## 🤝 联系方式

如有问题或建议，请通过以下方式联系:
- 📧 Email: your-email@example.com
- 💬 Issues: [GitHub Issues](https://github.com/your-repo/issues)

---

## ⭐ 致谢

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
