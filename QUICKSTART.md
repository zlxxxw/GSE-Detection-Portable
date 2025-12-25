# Portable Model - Quick Start

## 📦 What's Included

This is a minimal, portable package containing:
- ✅ Trained YOLOv11 model for Airport GSE detection
- ✅ Simple Python API (single file)
- ✅ Example scripts (image, video, API integration)
- ✅ Minimal dependencies

## 🚀 Usage (3 steps)

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Test detection:**
```bash
python detector.py your_image.jpg
```

3. **Integrate into your project:**
```python
from detector import GSEDetector

detector = GSEDetector(device="cuda")
detections = detector.detect(image)
```

## 📖 Full Documentation

See [README.md](README.md) for complete API documentation and examples.

## 🎯 Detected Classes

- 🚚 Galley_Truck (餐车)
- 🤖 Unmaned_GSE (无人GSE)
- 👷 Ground_Crew (地勤人员)
- ✈️ airplane (飞机)
