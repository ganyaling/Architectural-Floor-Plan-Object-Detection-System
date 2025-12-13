"""
PyTorch 原生 Faster R-CNN 训练脚本
用于 CubiCasa5K 数据集的目标检测训练
无需 MMDetection 复杂依赖
"""

import argparse
import os
import sys
from pathlib import Path
import json
import warnings
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import transforms, functional as F
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import json
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings('ignore')


class COCODetectionDataset(Dataset):
    """COCO 格式的目标检测数据集"""
    
    def __init__(self, img_dir, ann_file, transforms=None):
        """
        Args:
            img_dir: 图像目录路径（或父目录）
            ann_file: COCO 标注文件路径 (JSON)
            transforms: 数据增强转换
        """
        self.img_dir = Path(img_dir)
        self.coco = coco.COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        
        # 检查并确定实际的图像路径
        self._setup_img_paths()
    
    def _setup_img_paths(self):
        """设置图像路径 - 自动处理 cubicasa5k 的嵌套结构"""
        # cubicasa5k 的实际图像在 cubicasa5k/cubicasa5k/cubicasa5k 下（三层嵌套）
        possible_dirs = [
            self.img_dir,
            self.img_dir / 'cubicasa5k',
            self.img_dir / 'cubicasa5k' / 'cubicasa5k',
        ]
        
        for img_dir in possible_dirs:
            if img_dir.exists():
                # 检查是否能找到第一个图像
                test_file = self.coco.imgs[self.ids[0]]['file_name']
                # 清理文件名中的前缀
                test_file_clean = test_file.lstrip('/')
                if test_file_clean.startswith('kaggle/'):
                    test_file_clean = test_file_clean.replace('kaggle/input/cubicasa5k/cubicasa5k/cubicasa5k/', '', 1)
                
                # 转换正斜杠为反斜杠
                test_file_clean = test_file_clean.replace('/', '\\')
                test_path = img_dir / test_file_clean
                
                if test_path.exists():
                    self.img_dir = img_dir
                    print(f"✅ 使用图像目录: {self.img_dir}")
                    return
        
        # 如果找不到，使用原始目录并打印警告
        print(f"⚠️  警告: 无法自动定位图像目录，使用指定路径: {self.img_dir}")
    
    def __len__(self):
        return len(self.ids)
    
    def _get_img_path(self, file_name):
        """获取图像的完整路径"""
        # 清理 file_name 中的特殊前缀
        file_name = file_name.lstrip('/')
        
        # 处理 Kaggle 格式的路径
        if file_name.startswith('kaggle/'):
            file_name = file_name.replace('kaggle/input/cubicasa5k/cubicasa5k/cubicasa5k/', '', 1)
            if file_name.startswith('/'):
                file_name = file_name.lstrip('/')
        
        # 将路径中的正斜杠转换为系统斜杠
        file_name = file_name.replace('/', '\\')
        
        img_path = self.img_dir / file_name
        return img_path
    
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        
        # 获取图像路径
        img_path = self._get_img_path(img_info['file_name'])
        
        # 加载图像
        if not img_path.exists():
            print(f"\n❌ 图像文件不存在:")
            print(f"   原始名称: {img_info['file_name']}")
            print(f"   期望路径: {img_path}")
            print(f"   图像目录: {self.img_dir}")
            raise FileNotFoundError(f"图像文件不存在: {img_path}")
        
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # 自动缩放大图像以节省显存
        max_size = 1024  # 最大尺寸
        h, w = image.shape[:2]
        if w > max_size or h > max_size:
            scale = max_size / max(w, h)
            new_h, new_w = int(h * scale), int(w * scale)
            import cv2
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # 获取标注
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # 解析 bbox 和 category (格式: [x1, y1, x2, y2])
        bboxes = []
        class_labels = []
        img_h, img_w = image.shape[:2]
        
        for ann in anns:
            if ann['iscrowd']:
                continue
            x, y, w, h = ann['bbox']
            x1, y1, x2, y2 = x, y, x + w, y + h
            
            # 修剪 bbox 到图像范围内
            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(0, min(x2, img_w - 1))
            y2 = max(0, min(y2, img_h - 1))
            
            # 丢弃无效 bbox（太小或坐标无效）
            if x2 - x1 > 5 and y2 - y1 > 5 and x1 < x2 and y1 < y2:
                bboxes.append([x1, y1, x2, y2])
                class_labels.append(ann['category_id'])
        
        # 应用 Albumentations 变换
        if self.transforms:
            transformed = self.transforms(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            image = transformed['image']  # 已是 tensor
            bboxes = transformed['bboxes']
            class_labels = transformed['class_labels']
        
        # 转换为 tensor
        if len(bboxes) > 0:
            boxes = torch.as_tensor(bboxes, dtype=torch.float32)
            labels = torch.as_tensor(class_labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        
        # 如果没有 transforms，需要手动转张量
        if not self.transforms:
            image = transforms.ToTensor()(Image.fromarray(image))
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
        }
        
        return image, target


class Trainer:
    """训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        self.work_dir = Path(args.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # FP16 训练相关
        self.use_fp16 = torch.cuda.is_available() and args.use_fp16
        self.scaler = GradScaler() if self.use_fp16 else None
        
        print(f"✅ 使用设备: {self.device}")
        print(f"✅ 工作目录: {self.work_dir}")
        if self.use_fp16:
            print(f"✅ 启用 FP16 半精度训练")
    
    def get_model(self, num_classes=3):
        """获取预训练模型 - 使用迁移学习"""
        # 第一步：加载预训练模型（COCO 91类别）
        model = fasterrcnn_resnet50_fpn(pretrained=True, num_classes=91)
        
        # 第二步：替换最后的分类层，适配新的类别数
        # ROI Head 包含分类和边界框回归器
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        
        # 替换分类层：91类 → num_classes类
        model.roi_heads.box_predictor.cls_score = nn.Linear(in_features, num_classes)
        
        # 替换边界框回归层：91*4 → num_classes*4
        model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features, num_classes * 4)
        
        print(f"✅ 加载预训练 Faster R-CNN (ResNet50)")
        print(f"   - Backbone: ImageNet 预训练特征提取器")
        print(f"   - 分类头: 修改为 {num_classes} 个类别")
        print(f"   - 迁移学习：保留 Backbone，微调分类层")
        
        return model
    
    def collate_fn(self, batch):
        """自定义 collate 函数用于处理可变大小的 bbox"""
        return tuple(zip(*batch))
    
    def train_one_epoch(self, model, optimizer, train_loader, epoch):
        """训练一个 epoch (支持 FP16)"""
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.args.epochs}")
        for i, (images, targets) in enumerate(pbar):
            # 移到设备
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            
            # 使用 autocast 进行 FP16 前向传播
            if self.use_fp16:
                with autocast():
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                
                # FP16 反向传播
                self.scaler.scale(losses).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                # FP32 标准训练
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                losses.backward()
                optimizer.step()
            
            total_loss += losses.item()
            pbar.set_postfix({'loss': f'{losses.item():.4f}'})
            
            # 清理内存
            del images, targets, loss_dict, losses
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}: 平均损失 = {avg_loss:.4f}")
        return avg_loss
    
    @torch.no_grad()
    def evaluate(self, model, val_loader, ann_file):
        """验证模型"""
        model.eval()
        
        coco_gt = coco.COCO(ann_file)
        results = []
        
        print("评估中...")
        for images, targets in tqdm(val_loader):
            images = [img.to(self.device) for img in images]
            
            outputs = model(images)
            
            for output, target in zip(outputs, targets):
                img_id = target['image_id'].item()
                
                for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
                    x1, y1, x2, y2 = box.cpu().numpy()
                    w = x2 - x1
                    h = y2 - y1
                    
                    results.append({
                        'image_id': int(img_id),
                        'category_id': int(label.item()),
                        'bbox': [float(x1), float(y1), float(w), float(h)],
                        'score': float(score.item()),
                    })
        
        # 保存结果 (转换为 JSON 兼容的类型)
        results_file = self.work_dir / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f)
        
        # 评估 (需要添加 info 字段)
        # 如果原始 COCO 数据缺少 info，手动添加
        if 'info' not in coco_gt.dataset:
            coco_gt.dataset['info'] = {
                'description': 'CubiCasa5K detection results',
                'version': '1.0',
                'year': 2024
            }
        
        coco_dt = coco_gt.loadRes(str(results_file))
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        return coco_eval.stats[0]  # mAP
    
    def train(self):
        """执行训练"""
        print(f"\n{'='*70}")
        print(f"🚀 开始训练")
        print(f"{'='*70}")
        print(f"  骨干网络: {self.args.backbone}")
        print(f"  批量大小: {self.args.batch_size}")
        print(f"  训练轮数: {self.args.epochs}")
        print(f"  学习率: {self.args.lr}")
        print(f"{'='*70}\n")
        
        # 检查数据
        data_root = Path(self.args.data_root)
        coco_dir = data_root  # COCO JSON 所在目录
        # 图像实际在 cubicasa5k/cubicasa5k/cubicasa5k 下（三层嵌套）
        img_root = data_root.parent / 'cubicasa5k' / 'cubicasa5k'
        
        train_json = coco_dir / 'train_coco_pt.json'
        val_json = coco_dir / 'val_coco_pt.json'
        
        if not train_json.exists() or not val_json.exists():
            print(f"❌ 数据文件不存在!")
            print(f"   期望: {train_json}, {val_json}")
            sys.exit(1)
        
        if not img_root.exists():
            print(f"❌ 图像目录不存在!")
            print(f"   期望: {img_root}")
            print(f"   请确保原始图像在同一级目录下")
            sys.exit(1)
        
        print(f"✅ 使用 COCO 标注目录: {coco_dir}")
        print(f"✅ 使用图像目录: {img_root}\n")
        
        # 创建数据增强策略 (使用 Albumentations - 边界框感知)
        # ✨ Albumentations 原生支持 bbox 变换
        transform_train = A.Compose([
            # 几何变换 (自动处理 bbox)
            A.HorizontalFlip(p=0.5),                    # 水平翻转
            A.VerticalFlip(p=0.3),                      # 竖直翻转
            A.Rotate(limit=15, p=0.5, border_mode=0),  # 旋转 ±15度
            A.Affine(
                translate_percent=(0.05, 0.05),
                p=0.5
            ),                                           # 平移 5%
            
            # 色彩增强
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05,
                p=0.5
            ),                                           # 色彩抖动
            A.GaussianBlur(blur_limit=3, p=0.3),       # 高斯模糊
            
            # 标准化处理
            A.Normalize(
                mean=[123.675/255, 116.28/255, 103.53/255],
                std=[58.395/255, 57.12/255, 57.375/255],
                always_apply=True
            ),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',  # [x1, y1, x2, y2]
            min_visibility=0.3,   # 保留可见度 > 30% 的 bbox
            label_fields=['class_labels']
        ))
        
        # 验证集仅进行标准化处理（无增强）
        transform_val = A.Compose([
            A.Normalize(
                mean=[123.675/255, 116.28/255, 103.53/255],
                std=[58.395/255, 57.12/255, 57.375/255],
                always_apply=True
            ),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            min_visibility=0.3,
            label_fields=['class_labels']
        ))
        
        train_dataset = COCODetectionDataset(str(img_root), str(train_json), transform_train)
        val_dataset = COCODetectionDataset(str(img_root), str(val_json), transform_val)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=2
        )
        
        print(f"✅ 加载数据集:")
        print(f"   训练样本: {len(train_dataset)}")
        print(f"   验证样本: {len(val_dataset)}\n")
        
        # 获取模型
        # num_classes=3: 0=background (自动添加), 1=wall, 2=room
        model = self.get_model(num_classes=3)
        model.to(self.device)
        
        # 优化器
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.SGD(params, lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
        
        # 学习率调度 (Cosine Annealing - 更平滑的衰减)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.args.epochs,
            eta_min=self.args.lr * 0.01
        )
        
        # 训练循环
        best_ap = 0
        for epoch in range(1, self.args.epochs + 1):
            # 训练
            avg_loss = self.train_one_epoch(model, optimizer, train_loader, epoch)
            
            # 更新学习率
            scheduler.step()
            
            # 验证
            if epoch % self.args.val_interval == 0:
                ap = self.evaluate(model, val_loader, str(val_json))
                
                if ap > best_ap:
                    best_ap = ap
                    checkpoint = {
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'ap': ap,
                    }
                    torch.save(checkpoint, self.work_dir / 'best_model.pth')
                    print(f"✅ 保存最佳模型 (AP={ap:.4f})\n")
            
            # 定期保存检查点
            if epoch % 3 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(checkpoint, self.work_dir / f'checkpoint_epoch_{epoch}.pth')
        
        print(f"\n✅ 训练完成!")
        print(f"   最佳 AP: {best_ap:.4f}")
        print(f"   模型保存在: {self.work_dir}")


def main():
    parser = argparse.ArgumentParser(description='PyTorch Faster R-CNN 训练脚本')
    
    parser.add_argument(
        '--backbone',
        type=str,
        default='resnet50',
        choices=['resnet50'],
        help='骨干网络 (当前仅支持 resnet50)'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default=r"C:/Users/kawayi_yaling/.cache/kagglehub/datasets/qmarva/cubicasa5k/versions/4/cubicasa5k_coco",
        help='数据集根目录'
    )
    parser.add_argument(
        '--work-dir',
        type=str,
        default='./pytorch_detection_results',
        help='输出目录'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=2,
        help='批量大小 (默认 2, 大图像会自动缩放至 1024px)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=1,
        help='训练轮数'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.002,
        help='初始学习率'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU ID'
    )
    parser.add_argument(
        '--val-interval',
        type=int,
        default=1,
        help='验证间隔 (epochs)'
    )
    parser.add_argument(
        '--use-fp16',
        action='store_true',
        default=True,
        help='使用 FP16 半精度训练 (默认启用)'
    )
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    
    # 创建训练器并执行训练
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
