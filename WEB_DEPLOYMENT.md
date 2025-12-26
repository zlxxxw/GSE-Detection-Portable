# GSE检测模型 - Web端部署指南

## 📦 模型转换完成

您的 `gse_detection_v11.pt` 模型已成功转换为ONNX格式，可以在网页端使用！

### 📂 生成的文件

```
GSE_Detection_Portable/
├── onnx_model/
│   └── model.onnx              # 转换后的ONNX模型 (36.18 MB)
├── web_onnx_inference.html     # 网页端推理示例
├── convert_to_tfjs.py          # 模型转换脚本
└── WEB_DEPLOYMENT.md           # 本文档
```

## 🚀 快速开始

### 方案选择：ONNX Runtime Web（推荐）

我们采用 **ONNX Runtime Web** 而非TensorFlow.js，原因如下：

✅ **优势**
- 依赖简单，无需复杂的TensorFlow.js环境
- 性能优秀，支持WebGL和WebAssembly加速
- 原生支持YOLO ONNX模型，无需额外适配
- 模型文件更小（相比TensorFlow.js）

### 1. 本地测试

由于浏览器安全限制，需要通过HTTP服务器运行：

**方法一：使用Python HTTP服务器**
```bash
cd GSE_Detection_Portable
python -m http.server 8000
```

然后在浏览器打开：http://localhost:8000/web_onnx_inference.html

**方法二：使用VS Code Live Server扩展**
1. 安装 "Live Server" 扩展
2. 右键点击 `web_onnx_inference.html`
3. 选择 "Open with Live Server"

### 2. 使用说明

1. **加载模型**：页面打开时自动加载ONNX模型
2. **选择图片**：点击"选择图片"按钮上传测试图片
3. **开始检测**：点击"开始检测"按钮进行推理
4. **查看结果**：检测框会绘制在图片上，并显示类别和置信度

### 3. 部署到生产环境

#### 3.1 文件准备
需要部署的文件：
- `onnx_model/model.onnx` - ONNX模型文件
- `web_onnx_inference.html` - 前端页面（或集成到您的项目中）

#### 3.2 部署选项

**选项A：静态网站托管**
- GitHub Pages
- Netlify
- Vercel
- AWS S3 + CloudFront

**选项B：传统Web服务器**
- Nginx
- Apache
- IIS

#### 3.3 CORS配置

如果模型文件和网页在不同域名，需要配置CORS：

**Nginx配置示例：**
```nginx
location /models/ {
    add_header Access-Control-Allow-Origin *;
    add_header Access-Control-Allow-Methods 'GET, OPTIONS';
}
```

## 🔧 自定义与集成

### 修改模型路径

在 `web_onnx_inference.html` 中修改：
```javascript
const MODEL_CONFIG = {
    modelPath: './onnx_model/model.onnx', // 改为你的模型路径
    inputSize: 640,
    scoreThreshold: 0.25,
    iouThreshold: 0.45,
};
```

### 修改类别标签

根据您的训练数据修改 `CLASS_NAMES` 数组：
```javascript
const CLASS_NAMES = [
    '你的类别1',
    '你的类别2',
    // ...
];
```

### 集成到现有项目

1. **引入ONNX Runtime Web**
```html
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.0/dist/ort.min.js"></script>
```

2. **加载模型**
```javascript
const session = await ort.InferenceSession.create('path/to/model.onnx');
```

3. **执行推理**
```javascript
const inputTensor = new ort.Tensor('float32', inputData, [1, 3, 640, 640]);
const feeds = { images: inputTensor };
const results = await session.run(feeds);
```

## 📊 性能优化

### 1. 使用WebGL加速
```javascript
const options = {
    executionProviders: ['webgl'], // 优先使用WebGL
    graphOptimizationLevel: 'all'
};
const session = await ort.InferenceSession.create(modelPath, options);
```

### 2. 模型量化
当前模型已使用FP16量化，模型大小减少约50%。

### 3. 输入图片预处理
- 将图片缩放到640x640
- 转换为RGB格式
- 归一化到[0, 1]

## 🐛 常见问题

### Q1: 模型加载失败
**A:** 检查：
- 模型文件路径是否正确
- 是否通过HTTP服务器访问（不能直接打开HTML文件）
- 浏览器控制台是否有CORS错误

### Q2: 检测速度慢
**A:** 尝试：
- 确保使用WebGL加速
- 减小输入图片尺寸
- 使用更强大的设备/浏览器

### Q3: 检测结果不准确
**A:** 调整参数：
- `scoreThreshold`: 置信度阈值（降低以检测更多目标）
- `iouThreshold`: NMS IoU阈值（降低以减少重复检测）

### Q4: 在移动设备上运行
**A:** 
- 移动浏览器性能较弱，推理速度会较慢
- 考虑使用更小的模型或降低输入尺寸
- iOS Safari 和 Android Chrome 都支持

## 📱 移动端部署

### React Native
使用 `onnxruntime-react-native`:
```bash
npm install onnxruntime-react-native
```

### 微信小程序
需要使用 WeChat ML SDK 或后端API方式

## 🔄 重新转换模型

如果需要修改模型或重新转换：

```bash
# 激活环境
conda activate python311

# 运行转换脚本
python convert_to_tfjs.py
```

## 📖 参考资源

- [ONNX Runtime Web 文档](https://onnxruntime.ai/docs/tutorials/web/)
- [Ultralytics YOLO 文档](https://docs.ultralytics.com/)
- [ONNX 官方文档](https://onnx.ai/)

## 💡 下一步建议

1. **性能测试**：在目标设备上测试推理性能
2. **API封装**：将推理逻辑封装成API
3. **UI优化**：美化用户界面，添加更多功能
4. **监控部署**：添加性能监控和错误追踪

## 🎉 完成！

您的GSE检测模型现在可以在网页端运行了！

如有问题，请查看：
- 浏览器控制台的错误信息
- `web_onnx_inference.html` 中的注释
- ONNX Runtime Web 官方文档
