"""
FastAPI 后端服务 - 用于模型推理
支持上传图片并返回检测结果
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
import torch
import io
import json
from pathlib import Path
from typing import List, Dict
import uvicorn

from simple_inference import DetectionInference

# 初始化 FastAPI
app = FastAPI(
    title="建筑平面图目标检测 API",
    description="用于检测建筑平面图中的墙壁和房间",
    version="1.0.0"
)

# 添加 CORS 支持（允许前端跨域请求）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量：推理引擎
inferencer = None

# 模型配置
MODEL_CONFIG = {
    'checkpoint_path': './pytorch_detection_results/best_model.pth',
    'conf_threshold': 0.5,
    'device': 'cuda:0'
}


@app.on_event("startup")
async def startup_event():
    """应用启动时加载模型"""
    global inferencer
    try:
        print("🔄 正在加载模型...")
        inferencer = DetectionInference(
            checkpoint_path=MODEL_CONFIG['checkpoint_path'],
            device=MODEL_CONFIG['device'],
            conf_threshold=MODEL_CONFIG['conf_threshold']
        )
        print("✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    print("🔴 应用关闭")


@app.get("/")
async def root():
    """健康检查"""
    return {
        "status": "ok",
        "message": "建筑平面图目标检测 API 正在运行",
        "version": "1.0.0"
    }


@app.post("/detect")
async def detect_objects(file: UploadFile = File(...), conf_threshold: float = 0.5):
    """
    上传图片进行目标检测
    
    参数:
        file: 上传的图片文件 (PNG, JPG)
        conf_threshold: 置信度阈值 (0.0-1.0)
    
    返回:
        {
            "status": "success",
            "image_size": [width, height],
            "detections": [
                {
                    "bbox": [x1, y1, x2, y2],
                    "category": 1,
                    "category_name": "wall",
                    "confidence": 0.95
                },
                ...
            ],
            "summary": {
                "wall": 28,
                "room": 17
            }
        }
    """
    try:
        if inferencer is None:
            raise HTTPException(status_code=500, detail="模型未加载")
        
        # 检查文件类型
        if file.content_type not in ['image/png', 'image/jpeg', 'image/jpg']:
            raise HTTPException(status_code=400, detail="仅支持 PNG 或 JPG 格式")
        
        # 读取上传的文件
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # 保存临时文件以便推理
        temp_path = f"temp_{file.filename}"
        image.save(temp_path)
        
        # 执行推理
        results = inferencer.infer_single(temp_path, score_threshold=conf_threshold)
        
        # 统计检测结果
        summary = {}
        for det in results['detections']:
            cat_name = det['category_name']
            summary[cat_name] = summary.get(cat_name, 0) + 1
        
        # 删除临时文件
        Path(temp_path).unlink()
        
        return {
            "status": "success",
            "image_size": results['image_size'],
            "detections": results['detections'],
            "summary": summary,
            "total": len(results['detections'])
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"推理失败: {str(e)}")


@app.post("/detect-with-visualization")
async def detect_with_visualization(file: UploadFile = File(...), conf_threshold: float = 0.5):
    """
    上传图片进行检测并返回可视化结果
    
    返回: 带有检测框的图片文件
    """
    try:
        if inferencer is None:
            raise HTTPException(status_code=500, detail="模型未加载")
        
        # 检查文件类型
        if file.content_type not in ['image/png', 'image/jpeg', 'image/jpg']:
            raise HTTPException(status_code=400, detail="仅支持 PNG 或 JPG 格式")
        
        # 读取上传的文件
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # 保存临时输入文件
        temp_input = f"temp_input_{file.filename}"
        image.save(temp_input)
        
        # 绘制检测结果
        temp_output = f"temp_output_{file.filename}"
        inferencer.draw_predictions(temp_input, temp_output, conf_threshold)
        
        # 读取输出图片
        response = FileResponse(
            temp_output,
            media_type='image/png',
            filename=f"detected_{file.filename}"
        )
        
        # 清理临时文件
        Path(temp_input).unlink(missing_ok=True)
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"推理失败: {str(e)}")


@app.get("/model-info")
async def model_info():
    """获取模型信息"""
    return {
        "model_name": "Faster R-CNN + ResNet50",
        "classes": {
            0: "background",
            1: "wall",
            2: "room"
        },
        "checkpoint": MODEL_CONFIG['checkpoint_path'],
        "device": MODEL_CONFIG['device'],
        "conf_threshold": MODEL_CONFIG['conf_threshold']
    }


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "model_loaded": inferencer is not None
    }


if __name__ == "__main__":
    # 启动服务器
    # 使用 uvicorn 运行: python app.py
    # 或命令行: uvicorn app:app --host 0.0.0.0 --port 8000 --reload
    
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║       建筑平面图目标检测 API 服务                                  ║
    ╠════════════════════════════════════════════════════════════════╣
    ║  API 文档:   http://localhost:8000/docs                      ║
    ║  Swagger UI: http://localhost:8000/docs                      ║
    ║  ReDoc:      http://localhost:8000/redoc                     ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )
