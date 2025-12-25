# GSE Detection Portable - Quick Start Script
# 快速启动脚本

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  GSE Detection Portable System" -ForegroundColor Cyan
Write-Host "  机场地面设备检测系统" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查Python环境
$pythonCmd = "python"
if (Get-Command python -ErrorAction SilentlyContinue) {
    Write-Host " Python已找到" -ForegroundColor Green
} else {
    Write-Host " 未找到Python" -ForegroundColor Red
    exit 1
}

# 菜单
Write-Host ""
Write-Host "请选择运行模式:" -ForegroundColor Yellow
Write-Host "1. 摄像头实时检测（带跟踪，2560x1440@60fps）"
Write-Host "2. 摄像头实时检测（仅检测）"
Write-Host "3. 视频文件检测"
Write-Host "4. 图像检测"
Write-Host "5. 检查GPU状态"
Write-Host "Q. 退出"
Write-Host ""

$choice = Read-Host "请输入选项"

switch ($choice) {
    "1" {
        Write-Host "`n启动摄像头检测（带跟踪）..." -ForegroundColor Green
        python examples/video_detection.py 0 --enable-tracking --device cuda
    }
    "2" {
        Write-Host "`n启动摄像头检测（仅检测）..." -ForegroundColor Green
        python examples/video_detection.py 0 --device cuda
    }
    "3" {
        $videoPath = Read-Host "`n请输入视频文件路径"
        Write-Host "启动视频检测..." -ForegroundColor Green
        python examples/video_detection.py $videoPath --enable-tracking --device cuda
    }
    "4" {
        $imagePath = Read-Host "`n请输入图像文件路径"
        Write-Host "启动图像检测..." -ForegroundColor Green
        python examples/image_detection.py $imagePath
    }
    "5" {
        Write-Host "`n检查GPU状态..." -ForegroundColor Green
        python -c "import torch; print(f''CUDA可用: {torch.cuda.is_available()}'')"
    }
    "Q" {
        Write-Host "`n再见!" -ForegroundColor Yellow
        exit 0
    }
    default {
        Write-Host "`n无效选项" -ForegroundColor Red
        exit 1
    }
}
