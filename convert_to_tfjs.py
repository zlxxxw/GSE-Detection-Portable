"""
模型转换脚本: PyTorch (.pt) → ONNX (适用于Web端)

由于TensorFlow.js的依赖复杂性，我们采用更简单的方案：
1. 将YOLO模型导出为ONNX格式（FP16量化）
2. 在Web端使用onnxruntime-web (支持WebGL/WebAssembly)

这种方案的优势:
- 无需TensorFlow.js复杂的依赖
- ONNX Runtime Web性能优秀
- 更容易部署和维护

依赖安装:
pip install ultralytics onnx onnxsim

使用方法:
python convert_to_tfjs.py
"""

import os
import sys
from pathlib import Path
import shutil

def check_dependencies():
    """检查必要的依赖包"""
    required_packages = {
        'ultralytics': 'YOLO模型操作',
        'onnx': 'ONNX模型处理',
    }
    
    optional_packages = {
        'onnxsim': 'ONNX模型简化（可选）'
    }
    
    missing_packages = []
    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {package} - {description}")
        except ImportError:
            print(f"✗ {package} - {description} (未安装)")
            missing_packages.append(package)
    
    # 检查可选包
    for package, description in optional_packages.items():
        try:
            __import__(package)
            print(f"✓ {package} - {description}")
        except ImportError:
            print(f"○ {package} - {description} (未安装，推荐安装)")
    
    if missing_packages:
        print(f"\n缺少依赖包，请运行:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    return True

def convert_yolo_to_onnx(model_path, output_dir, img_size=640):
    """步骤1: 将YOLO模型转换为ONNX（带FP16量化）"""
    print("\n" + "="*60)
    print("步骤 1/2: PyTorch YOLO → ONNX (FP16)")
    print("="*60)
    
    from ultralytics import YOLO
    
    # 加载YOLO模型
    model = YOLO(model_path)
    print(f"✓ 加载模型: {model_path}")
    
    # 导出为ONNX格式
    onnx_path = output_dir / "model.onnx"
    model.export(
        format='onnx',
        imgsz=img_size,
        simplify=True,  # 简化ONNX模型
        opset=12,  # ONNX opset版本
        half=True,  # 使用FP16量化
    )
    
    # YOLO导出的文件名格式为: model_name.onnx
    model_name = Path(model_path).stem
    exported_onnx = Path(model_path).parent / f"{model_name}.onnx"
    
    # 如果导出的文件存在，移动到输出目录
    if exported_onnx.exists():
        if onnx_path.exists():
            onnx_path.unlink()
        shutil.move(str(exported_onnx), str(onnx_path))
    else:
        # 检查当前目录
        current_dir_onnx = Path.cwd() / f"{model_name}.onnx"
        if current_dir_onnx.exists():
            if onnx_path.exists():
                onnx_path.unlink()
            shutil.move(str(current_dir_onnx), str(onnx_path))
    
    print(f"✓ ONNX模型已保存: {onnx_path}")
    print(f"  - 量化格式: FP16 (半精度)")
    return onnx_path

def optimize_onnx(onnx_path, output_dir):
    """步骤2: 优化ONNX模型（可选）"""
    print("\n" + "="*60)
    print("步骤 2/2: 优化ONNX模型")
    print("="*60)
    
    try:
        import onnxsim
        import onnx
        
        # 加载ONNX模型
        model = onnx.load(str(onnx_path))
        print(f"✓ 加载ONNX模型: {onnx_path}")
        
        # 简化模型
        model_simp, check = onnxsim.simplify(model)
        
        if check:
            # 保存简化后的模型
            optimized_path = output_dir / "model_optimized.onnx"
            onnx.save(model_simp, str(optimized_path))
            print(f"✓ 优化模型已保存: {optimized_path}")
            return optimized_path
        else:
            print("⚠ 模型简化失败，使用原始模型")
            return onnx_path
            
    except ImportError:
        print("○ onnxsim未安装，跳过优化步骤")
        return onnx_path
    except Exception as e:
        print(f"⚠ 优化过程出错: {e}")
        print("  使用原始模型")
        return onnx_path

def get_model_info(onnx_path):
    """获取并显示模型信息"""
    import onnx
    
    try:
        model = onnx.load(str(onnx_path))
        
        print("\n" + "="*60)
        print("模型信息")
        print("="*60)
        print(f"格式: ONNX (适用于onnxruntime-web)")
        print(f"位置: {onnx_path}")
        
        # 计算模型大小
        size_mb = onnx_path.stat().st_size / (1024*1024)
        print(f"模型大小: {size_mb:.2f} MB")
        
        # 获取输入输出信息
        graph = model.graph
        if graph.input:
            inp = graph.input[0]
            print(f"\n输入信息:")
            print(f"  名称: {inp.name}")
            dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            print(f"  形状: {dims}")
        
        print(f"\n网页端加载方式 (onnxruntime-web):")
        print(f"  const session = await ort.InferenceSession.create('model.onnx');")
        
    except Exception as e:
        print(f"⚠ 无法读取模型信息: {e}")

def main():
    """主函数"""
    print("="*60)
    print("GSE检测模型转换工具")
    print("PyTorch YOLO → ONNX (FP16) for Web")
    print("="*60)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 设置路径
    current_dir = Path(__file__).parent
    model_path = current_dir / "weights" / "gse_detection_v11.pt"
    output_dir = current_dir / "onnx_model"
    
    # 检查模型文件
    if not model_path.exists():
        print(f"\n错误: 找不到模型文件: {model_path}")
        sys.exit(1)
    
    # 创建输出目录
    output_dir.mkdir(exist_ok=True)
    print(f"\n输出目录: {output_dir}")
    
    try:
        # 步骤1: PyTorch YOLO → ONNX (FP16)
        onnx_path = convert_yolo_to_onnx(model_path, output_dir)
        
        # 步骤2: 优化ONNX模型（可选）
        final_path = optimize_onnx(onnx_path, output_dir)
        
        # 显示模型信息
        get_model_info(final_path)
        
        print("\n" + "="*60)
        print("转换完成! ✓")
        print("="*60)
        print(f"\nONNX模型位置: {final_path}")
        print("\n后续步骤:")
        print("1. 将ONNX模型文件部署到Web服务器")
        print("2. 在HTML中使用onnxruntime-web加载模型")
        print("3. 参考web_onnx_inference.html进行推理")
        print("\n推荐使用onnxruntime-web的原因:")
        print("  ✓ 依赖简单，无需复杂的TensorFlow.js环境")
        print("  ✓ 性能优秀，支持WebGL和WebAssembly加速")
        print("  ✓ 原生支持YOLO ONNX模型")
        print("  ✓ 模型更小，FP16量化减少50%体积")
        
    except Exception as e:
        print(f"\n错误: 转换失败")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
