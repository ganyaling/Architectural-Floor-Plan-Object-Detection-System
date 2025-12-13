@echo off
REM ==========================================
REM 快速启动脚本 - 建筑平面图检测系统
REM ==========================================

echo.
echo ========================================
echo   建筑平面图目标检测 - 快速启动
echo ========================================
echo.

REM 获取当前脚本所在目录
cd /d "%~dp0"

REM 激活虚拟环境
call conda activate mmenv
if errorlevel 1 (
    echo [ERROR] 无法激活 conda 环境 mmenv
    echo 请确保已安装 conda 并创建了 mmenv 环境
    pause
    exit /b 1
)

echo [1/4] 检查 Python 环境...
python -c "import sys; print(f'Python 版本: {sys.version.split()[0]}')"

echo.
echo [2/4] 检查依赖包...
python -c "import torch; print(f'PyTorch 版本: {torch.__version__}')" 2>nul || (
    echo [WARN] PyTorch 未安装，尝试安装...
    pip install torch torchvision
)

python -c "import fastapi; import uvicorn; print('FastAPI 依赖已安装')" 2>nul || (
    echo [WARN] FastAPI 未安装，尝试安装...
    pip install fastapi uvicorn python-multipart
)

REM 检查模型文件
echo.
echo [3/4] 检查模型文件...
if not exist "pytorch_detection_results\best_model.pth" (
    echo [ERROR] 未找到模型文件: pytorch_detection_results\best_model.pth
    echo 请先训练模型或复制模型文件到此目录
    pause
    exit /b 1
)
echo [OK] 模型文件已就位

REM 检查前端文件
echo.
if not exist "index.html" (
    echo [ERROR] 未找到前端文件: index.html
    pause
    exit /b 1
)
echo [OK] 前端文件已就位

echo.
echo ========================================
echo   启动后端和前端服务...
echo ========================================
echo.
echo 前端地址: http://localhost:8001
echo 后端 API: http://localhost:8000
echo API 文档: http://localhost:8000/docs
echo.
echo 在浏览器中打开: http://localhost:8001
echo 按 Ctrl+C 停止服务
echo.

REM 在后台启动前端 HTTP 服务器
start /B python -m http.server 8001 > nul 2>&1

REM 等待前端服务启动
timeout /t 2 /nobreak

REM 打开浏览器
start http://localhost:8001

REM 启动后端
python app.py

pause
