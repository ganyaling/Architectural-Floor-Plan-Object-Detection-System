#!/usr/bin/env python3
"""
快速启动脚本 - 建筑平面图检测系统
支持后端服务和前端服务的一键启动
"""

import os
import sys
import subprocess
import webbrowser
import time
import argparse
from pathlib import Path

def check_model_exists():
    """检查模型文件是否存在"""
    model_path = Path("pytorch_detection_results/best_model.pth")
    if not model_path.exists():
        print(f"[ERROR] 模型文件不存在: {model_path}")
        print("请先运行训练脚本生成模型")
        return False
    print(f"[OK] 模型文件已就位: {model_path}")
    return True

def check_frontend_exists():
    """检查前端文件是否存在"""
    if not Path("index.html").exists():
        print("[ERROR] 前端文件不存在: index.html")
        return False
    print("[OK] 前端文件已就位: index.html")
    return True

def install_dependencies():
    """安装必要的依赖"""
    print("\n[*] 正在检查依赖...")
    
    try:
        import torch
        print(f"[OK] PyTorch {torch.__version__}")
    except ImportError:
        print("[WARN] PyTorch 未安装，请运行: pip install torch torchvision")
        return False
    
    try:
        import fastapi
        import uvicorn
        print("[OK] FastAPI & Uvicorn 已安装")
    except ImportError:
        print("[*] 安装 FastAPI & Uvicorn...")
        subprocess.run([sys.executable, "-m", "pip", "install", 
                       "fastapi", "uvicorn", "python-multipart"],
                      check=True)
    
    return True

def start_backend(host="127.0.0.1", port=8000, reload=False):
    """启动后端服务"""
    print(f"\n[*] 启动后端服务...")
    print(f"    地址: http://{host}:{port}")
    print(f"    API 文档: http://{host}:{port}/docs")
    
    cmd = [sys.executable, "-m", "uvicorn", 
           "app:app", 
           f"--host", host,
           f"--port", str(port)]
    
    if reload:
        cmd.append("--reload")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[*] 后端服务已停止")

def start_frontend(port=8001):
    """启动前端服务"""
    print(f"\n[*] 启动前端服务...")
    print(f"    地址: http://127.0.0.1:{port}")
    
    cmd = [sys.executable, "-m", "http.server", str(port)]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[*] 前端服务已停止")

def open_browser(url, delay=2):
    """打开浏览器"""
    print(f"\n[*] 将在 {delay} 秒后打开浏览器: {url}")
    time.sleep(delay)
    try:
        webbrowser.open(url)
        print("[OK] 浏览器已打开")
    except Exception as e:
        print(f"[WARN] 无法自动打开浏览器: {e}")
        print(f"请手动访问: {url}")

def main():
    parser = argparse.ArgumentParser(
        description="建筑平面图检测系统 - 快速启动脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python start.py                    # 启动后端服务
  python start.py --mode frontend    # 启动前端服务
  python start.py --mode both        # 同时启动前端和后端
  python start.py --host 0.0.0.0     # 监听所有网卡
  python start.py --reload           # 启用热重载模式
        """
    )
    
    parser.add_argument("--mode", 
                       choices=["backend", "frontend", "both"],
                       default="backend",
                       help="启动模式 (默认: backend)")
    
    parser.add_argument("--host",
                       default="127.0.0.1",
                       help="后端服务主机 (默认: 127.0.0.1)")
    
    parser.add_argument("--port",
                       type=int,
                       default=8000,
                       help="后端服务端口 (默认: 8000)")
    
    parser.add_argument("--frontend-port",
                       type=int,
                       default=8001,
                       help="前端服务端口 (默认: 8001)")
    
    parser.add_argument("--reload",
                       action="store_true",
                       help="启用热重载模式 (开发环境)")
    
    parser.add_argument("--open-browser",
                       action="store_true",
                       help="启动后自动打开浏览器")
    
    args = parser.parse_args()
    
    # 改变工作目录到脚本所在目录
    os.chdir(Path(__file__).parent)
    
    # 检查文件
    if not check_model_exists():
        return 1
    
    if args.mode in ["frontend", "both"]:
        if not check_frontend_exists():
            return 1
    
    # 安装依赖
    if not install_dependencies():
        return 1
    
    # 启动服务
    print("\n" + "="*50)
    print("建筑平面图目标检测系统")
    print("="*50)
    
    if args.mode == "backend":
        start_backend(args.host, args.port, args.reload)
    
    elif args.mode == "frontend":
        if args.open_browser:
            open_browser(f"http://127.0.0.1:{args.frontend_port}", delay=1)
        start_frontend(args.frontend_port)
    
    elif args.mode == "both":
        # 在这种模式下，我们需要使用多进程
        import threading
        
        backend_thread = threading.Thread(
            target=start_backend,
            args=(args.host, args.port, args.reload),
            daemon=True
        )
        
        frontend_thread = threading.Thread(
            target=start_frontend,
            args=(args.frontend_port,),
            daemon=True
        )
        
        print("\n[*] 启动所有服务...")
        backend_thread.start()
        frontend_thread.start()
        
        if args.open_browser:
            open_browser(f"http://127.0.0.1:{args.frontend_port}", delay=3)
        
        # 保持主线程运行
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[*] 所有服务已停止")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
