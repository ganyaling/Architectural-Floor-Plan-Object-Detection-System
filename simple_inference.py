"""
PyTorch Faster R-CNN 推理脚本
用于单张或批量目标检测推理
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import json

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import transforms
from tqdm import tqdm
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval


class DetectionInference:
    """目标检测推理工具"""
    
    # 类别 ID 到名称的映射
    CLASSES = {
        0: 'background',
        1: 'wall',
        2: 'room'
    }
    
    # 颜色映射
    COLORS = {
        'wall': 'red',
        'room': 'blue',
        'background': 'green'
    }
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda:0', conf_threshold: float = 0.5):
        """
        初始化推理引擎
        
        Args:
            checkpoint_path: 模型检查点路径
            device: 设备（'cuda:0' 或 'cpu'）
            conf_threshold: 置信度阈值
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.conf_threshold = conf_threshold
        
        print(f"✅ 使用设备: {self.device}")
        print(f"✅ 置信度阈值: {conf_threshold}")
        
        # 加载模型
        self.model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=3)
        
        # 加载检查点
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ 模型已加载: {checkpoint_path}")
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """预处理图像"""
        transform = transforms.ToTensor()
        return transform(image).to(self.device)
    
    def infer_single(self, image_path: str, score_threshold: float = None) -> Dict:
        """
        单张图像推理
        
        Args:
            image_path: 图像路径
            score_threshold: 分数阈值（如果为 None 使用默认值）
        
        Returns:
            推理结果字典
        """
        if score_threshold is None:
            score_threshold = self.conf_threshold
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        img_tensor = self.preprocess(image)
        
        # 推理
        with torch.no_grad():
            predictions = self.model([img_tensor])
        
        pred = predictions[0]
        
        # 筛选高置信度检测
        mask = pred['scores'] >= score_threshold
        
        results = {
            'image_path': str(image_path),
            'image_size': image.size,  # (width, height)
            'detections': []
        }
        
        for box, label, score in zip(
            pred['boxes'][mask],
            pred['labels'][mask],
            pred['scores'][mask]
        ):
            x1, y1, x2, y2 = box.cpu().numpy().astype(float)
            label_id = label.item()
            confidence = score.item()
            
            results['detections'].append({
                'bbox': [x1, y1, x2, y2],
                'category': label_id,
                'category_name': self.CLASSES.get(label_id, 'unknown'),
                'confidence': confidence,
                'width': x2 - x1,
                'height': y2 - y1
            })
        
        return results
    
    def infer_batch(self, image_dir: str, output_json: str = None) -> List[Dict]:
        """
        批量推理
        
        Args:
            image_dir: 图像目录
            output_json: 结果保存 JSON 文件路径
        
        Returns:
            推理结果列表
        """
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob('**/*.png')) + list(image_dir.glob('**/*.jpg'))
        
        print(f"\n🔍 找到 {len(image_files)} 张图像")
        
        all_results = []
        
        for img_path in tqdm(image_files, desc="推理中"):
            try:
                results = self.infer_single(str(img_path))
                all_results.append(results)
            except Exception as e:
                print(f"⚠️  处理失败: {img_path} - {e}")
        
        # 保存结果
        if output_json:
            with open(output_json, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"✅ 结果已保存: {output_json}")
        
        return all_results
    
    def draw_predictions(self, image_path: str, output_path: str = None, 
                        score_threshold: float = None) -> Image.Image:
        """
        绘制检测结果
        
        Args:
            image_path: 图像路径
            output_path: 输出图像路径（如果为 None 则不保存）
            score_threshold: 分数阈值
        
        Returns:
            绘制后的图像
        """
        # 推理
        results = self.infer_single(image_path, score_threshold)
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        # 尝试加载字体
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # 绘制检测框
        for det in results['detections']:
            x1, y1, x2, y2 = det['bbox']
            label = det['category_name']
            confidence = det['confidence']
            color = self.COLORS.get(label, 'yellow')
            
            # 绘制矩形
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # 绘制标签
            text = f"{label} {confidence:.2f}"
            text_bbox = draw.textbbox((x1, y1), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # 标签背景
            draw.rectangle([x1, y1 - text_height - 5, x1 + text_width + 5, y1], 
                          fill=color)
            # 标签文本
            draw.text((x1 + 2, y1 - text_height - 3), text, fill='white', font=font)
        
        # 保存结果
        if output_path:
            image.save(output_path)
            print(f"✅ 可视化结果已保存: {output_path}")
        
        return image
    
    def evaluate_on_coco(self, coco_json_path: str, image_dir: str) -> Dict:
        """
        在 COCO 数据集上评估
        
        Args:
            coco_json_path: COCO 格式的标注文件
            image_dir: 图像目录
        
        Returns:
            评估指标
        """
        print("\n📊 在 COCO 数据集上评估...")
        
        # 加载 COCO 标注
        coco_gt = coco.COCO(coco_json_path)
        
        # 推理生成预测结果
        results = []
        image_ids = coco_gt.getImgIds()
        
        for img_id in tqdm(image_ids, desc="推理中"):
            img_info = coco_gt.loadImgs(img_id)[0]
            img_path = os.path.join(image_dir, img_info['file_name'])
            
            try:
                inference_result = self.infer_single(img_path)
                
                for det in inference_result['detections']:
                    x1, y1, x2, y2 = det['bbox']
                    w = x2 - x1
                    h = y2 - y1
                    
                    results.append({
                        'image_id': img_id,
                        'category_id': det['category'],
                        'bbox': [x1, y1, w, h],
                        'score': det['confidence']
                    })
            except Exception as e:
                print(f"⚠️  处理失败: {img_path} - {e}")
        
        # COCO 评估
        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        
        print("\n" + "=" * 70)
        coco_eval.summarize()
        print("=" * 70)
        
        return {
            'mAP': coco_eval.stats[0],
            'mAP_50': coco_eval.stats[1],
            'mAP_75': coco_eval.stats[2],
            'mAP_small': coco_eval.stats[3],
            'mAP_medium': coco_eval.stats[4],
            'mAP_large': coco_eval.stats[5],
        }


def main():
    parser = argparse.ArgumentParser(description='目标检测推理脚本')
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='模型检查点路径'
    )
    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help='单张图像推理路径'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        default=None,
        help='批量推理的图像目录'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./inference_results',
        help='输出目录'
    )
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.5,
        help='置信度阈值'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU ID'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='保存可视化结果'
    )
    parser.add_argument(
        '--evaluate',
        type=str,
        default=None,
        help='COCO 标注文件路径（用于评估）'
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 初始化推理引擎
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    inferencer = DetectionInference(args.checkpoint, device, args.conf_threshold)
    
    # 单张图像推理
    if args.image:
        print(f"\n🖼️  推理单张图像: {args.image}")
        results = inferencer.infer_single(args.image)
        
        print(f"\n📊 检测到 {len(results['detections'])} 个目标:")
        for det in results['detections']:
            print(f"  - {det['category_name']}: {det['confidence']:.2%}")
        
        # 保存结果
        output_json = os.path.join(args.output_dir, 'single_result.json')
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ 结果已保存: {output_json}")
        
        # 可视化
        if args.visualize:
            output_image = os.path.join(args.output_dir, 'single_result_viz.png')
            inferencer.draw_predictions(args.image, output_image)
    
    # 批量推理
    if args.image_dir:
        print(f"\n📁 批量推理目录: {args.image_dir}")
        output_json = os.path.join(args.output_dir, 'batch_results.json')
        all_results = inferencer.infer_batch(args.image_dir, output_json)
        
        total_detections = sum(len(r['detections']) for r in all_results)
        print(f"\n✅ 总共检测到 {total_detections} 个目标")
    
    # 在 COCO 数据集上评估
    if args.evaluate:
        image_dir = args.image_dir or args.image or args.output_dir
        inferencer.evaluate_on_coco(args.evaluate, image_dir)
    
    print(f"\n✅ 推理完成！结果保存在: {args.output_dir}")


if __name__ == '__main__':
    main()
