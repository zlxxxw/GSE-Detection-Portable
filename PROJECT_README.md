# GSE Detection Portable - 独立工程版

**机场地面设备检测系统 - 可移植独立版本**

这是一个完全独立的YOLO检测工程，无需依赖原始训练项目，可直接部署使用。

---

##  项目特点

 **完全独立** - 包含所有必需组件，无外部依赖  
 **功能完整** - 支持检测、跟踪、高分辨率、畸变矫正  
 **即插即用** - 3个命令即可开始使用  
 **生产就绪** - 优化的性能和稳定性  

---

##  项目结构

```
GSE_Detection_Portable/
 config.py              # 配置文件（类别、颜色、参数）
 detector.py            # 核心检测器
 byte_tracker.py        # ByteTrack跟踪算法
 camera_io.py           # 摄像头I/O控制
 requirements.txt       # 依赖清单
 start.ps1             # 快速启动脚本（Windows）
 .gitignore            # Git忽略文件
 README.md             # 本文档
 QUICKSTART.md         # 快速开始指南
 weights/
    gse_detection_v11.pt   # 训练好的模型
 examples/
     image_detection.py      # 图像检测示例
     video_detection.py      # 视频/摄像头检测（完整功能）
     api_integration.py      # API集成示例
```

---

##  快速开始

### 方法1：使用启动脚本（推荐）

```powershell
# Windows
.\start.ps1
```

### 方法2：直接命令

**1. 安装依赖**
```bash
pip install -r requirements.txt
```

**2. 摄像头实时检测（完整功能）**
```bash
python examples/video_detection.py 0 --enable-tracking --device cuda
```

**3. 图像检测**
```bash
python examples/image_detection.py your_image.jpg
```

---

##  配置说明

### 默认摄像头参数

在 `video_detection.py` 中：
- 分辨率：2560x1440
- 帧率：60 FPS
- 格式：MJPG
- 后端：dshow (Windows)

### 修改摄像头参数

```bash
python examples/video_detection.py 0 \
  --width 1920 \
  --height 1080 \
  --fps 30 \
  --device cuda
```

### 修改检测参数

编辑 `config.py`:
```python
CONFIDENCE_THRESHOLD = 0.25  # 置信度阈值
INPUT_SIZE = 1280            # 推理尺寸
DEVICE = "cuda"              # 设备
```

---

##  功能特性

### 1. 高性能检测
-  GPU加速（CUDA）
-  高分辨率支持（最高2560x1440）
-  实时处理（26+ FPS @ 2560x1440）

### 2. ByteTrack跟踪
-  多目标跟踪
-  ID持续性
-  轨迹预测
-  实时切换（T键）

### 3. 图像处理
-  畸变矫正（需标定文件）
-  自动缩放显示
-  实时FPS统计

### 4. 交互控制
- Q - 退出
- S - 保存当前帧
- U - 切换畸变矫正
- T - 切换跟踪

---

##  检测类别

| ID | 类别 | 颜色 | 说明 |
|----|------|------|------|
| 0 | Galley_Truck | 红色 | 餐车 |
| 1 | Unmaned_GSE | 红色 | 无人地面设备 |
| 2 | Ground_Crew | 蓝色 | 地勤人员 |
| 3 | airplane | 绿色 | 飞机 |

---

##  API使用

### Python API

```python
from detector import GSEDetector
import cv2

# 初始化
detector = GSEDetector(device="cuda")

# 读取图像
image = cv2.imread("test.jpg")

# 检测
detections = detector.detect(image)

# 绘制结果
result = detector.draw_results(image, detections)

# 保存
cv2.imwrite("result.jpg", result)
```

### 高级使用

```python
# 类别过滤
airplanes = detector.filter_by_class(
    detections, 
    class_names=["airplane"]
)

# 统计信息
stats = detector.get_statistics(detections)
print(stats)  # {'airplane': 2, 'Ground_Crew': 5}
```

---

##  系统要求

### 最低要求
- Python 3.8+
- 4GB RAM
- Windows 10/11
- CPU推理支持

### 推荐配置
- Python 3.11
- 16GB RAM
- NVIDIA GPU（8GB+ VRAM）
- CUDA 11.8+
- 高分辨率USB摄像头

### 摄像头要求

**推荐配置：**
- 分辨率： 1920x1080
- 帧率： 30 FPS
- 格式：MJPG
- 视角：俯视角度（15-45）

**最佳配置：**
- 分辨率：2560x1440
- 帧率：60 FPS
- 接口：USB 3.0+

---

##  性能基准

| 设备 | 分辨率 | 推理FPS | 显示FPS |
|------|--------|---------|---------|
| RTX A4000 | 2560x1440 | 60+ | 26+ |
| RTX 3060 | 1920x1080 | 70+ | 40+ |
| i7-12700 (CPU) | 1280x720 | 15+ | 15+ |

---

##  故障排除

### 1. CUDA不可用

```bash
# 检查CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 使用CPU（如无GPU）
python examples/video_detection.py 0 --device cpu
```

### 2. 摄像头无法打开

```bash
# 尝试不同后端
python examples/video_detection.py 0 --backend msmf

# 降低分辨率
python examples/video_detection.py 0 --width 1920 --height 1080
```

### 3. 低FPS

```bash
# 使用较小推理尺寸
python examples/video_detection.py 0 --model-size 640

# 降低分辨率
python examples/video_detection.py 0 --width 1280 --height 720
```

---

##  版本信息

**当前版本：** v1.0.0  
**发布日期：** 2025-12-25  
**模型版本：** YOLOv11s  
**训练数据：** 171张机场场景图像  

---

##  更新日志

### v1.0.0 (2025-12-25)
-  初始独立版本发布
-  完整功能移植
-  ByteTrack跟踪集成
-  高分辨率摄像头支持
-  畸变矫正功能
-  启动脚本和文档

---

##  许可证

MIT License

---

##  支持

如有问题或建议，请联系项目维护者。

---

##  重要提示

1. **场景限制**：此模型专为机场地勤场景训练，其他场景效果可能下降
2. **视角要求**：需要俯视角度（15-45）以获得最佳效果
3. **分辨率影响**：建议使用1920x1080分辨率
4. **GPU推荐**：强烈建议使用GPU以获得最佳性能

---

**Enjoy detecting! **
