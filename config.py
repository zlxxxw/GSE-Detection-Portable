"""
机场GSE检测模型配置文件
Airport GSE Detection Model Configuration
"""

# 类别配置
CLASS_NAMES = {
    0: "Galley_Truck",      # 餐车
    1: "Unmaned_GSE",       # 无人地面设备
    2: "Ground_Crew",       # 地勤人员
    3: "airplane"           # 飞机
}

# 类别颜色 (BGR格式)
CLASS_COLORS = {
    0: (0, 0, 255),      # 餐车 - 红色
    1: (0, 0, 255),      # 无人GSE - 红色
    2: (255, 0, 0),      # 地勤人员 - 蓝色
    3: (0, 255, 0)       # 飞机 - 绿色
}

# 模型配置
MODEL_PATH = "weights/gse_detection_v11.pt"
INPUT_SIZE = 1280          # 推理尺寸
CONFIDENCE_THRESHOLD = 0.25  # 置信度阈值

# 设备配置
DEVICE = "cuda"  # 可选: "cuda", "cpu", "mps"

# 类别中文名称（可选）
CLASS_NAMES_CN = {
    0: "餐车",
    1: "无人地面设备",
    2: "地勤人员",
    3: "飞机"
}
