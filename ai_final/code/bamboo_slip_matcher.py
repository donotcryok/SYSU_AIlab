"""
竹简缀合识别系统
用于识别竹简是否完整以及残片之间是否可以缀合
优化版 - 基于ResNet50 + 注意力机制 + 数据增强
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
import json
import random
from tqdm import tqdm
import logging
import re

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BambooSlipImageProcessor:
    """竹简图像处理器 - 优化版"""
    
    def __init__(self):
        self.edge_threshold = 50
        self.min_contour_area = 100
    
    def preprocess_bamboo_image(self, image_path, target_size=(256, 512), extract_key_regions=True):
        """
        竹简图像预处理 - 包含关键区域提取和边缘增强
        """
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        # 转换为RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 调整大小
        image = cv2.resize(image, target_size)
        height, width = image.shape[:2]
        
        if extract_key_regions:
            # 关键区域提取：只保留上下1/3区域（断口区域）
            crop_height = height // 3
            top_region = image[:crop_height, :]  # 上断口
            bottom_region = image[-crop_height:, :]  # 下断口
            # 重新组合关键区域
            image = np.vstack([top_region, bottom_region])
        
        # 边缘特征增强
        image = self._enhance_edge_features(image)
        
        # 归一化
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def _enhance_edge_features(self, image):
        """使用Sobel算子增强边缘特征"""
        # 转为灰度图用于边缘检测
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Sobel边缘检测
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # 计算边缘幅值
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_magnitude = (edge_magnitude / edge_magnitude.max() * 255).astype(np.uint8)
        
        # 转换回3通道
        edge_magnitude_3ch = cv2.cvtColor(edge_magnitude, cv2.COLOR_GRAY2RGB)
        
        # 与原图加权融合
        enhanced_image = cv2.addWeighted(image, 0.7, edge_magnitude_3ch, 0.3, 0)
        
        return enhanced_image
    
    def detect_completeness(self, image_path):
        """
        检测竹简是否完整
        返回: {'is_complete': bool, 'break_positions': list, 'width': float}
        """
        image = cv2.imread(str(image_path))
        if image is None:
            return {'is_complete': False, 'break_positions': [], 'width': 0}
        
        # 转为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 形态学操作
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'is_complete': False, 'break_positions': [], 'width': 0}
        
        # 找最大轮廓（假设为竹简主体）
        main_contour = max(contours, key=cv2.contourArea)
        
        # 获取边界矩形
        x, y, w, h = cv2.boundingRect(main_contour)
        
        # 分析边缘特征
        break_positions = self._analyze_breaks(gray, x, y, w, h)
        
        # 判断是否完整（简化版判断）
        is_complete = len(break_positions) == 0
        
        return {
            'is_complete': is_complete,
            'break_positions': break_positions,
            'width': w,
            'height': h,
            'contour_area': cv2.contourArea(main_contour)
        }
    
    def _analyze_breaks(self, gray_image, x, y, w, h):
        """分析断口位置"""
        breaks = []
        
        # 检查上下边缘的不规则性
        top_edge = gray_image[y:y+10, x:x+w]
        bottom_edge = gray_image[y+h-10:y+h, x:x+w]
        left_edge = gray_image[y:y+h, x:x+10]
        right_edge = gray_image[y:y+h, x+w-10:x+w]
        
        # 简化的断口检测（基于边缘变化）
        if self._is_irregular_edge(top_edge):
            breaks.append('top')
        if self._is_irregular_edge(bottom_edge):
            breaks.append('bottom')
        if self._is_irregular_edge(left_edge):
            breaks.append('left')
        if self._is_irregular_edge(right_edge):
            breaks.append('right')
        
        return breaks
    
    def _is_irregular_edge(self, edge_region):
        """判断边缘是否不规则（存在断口）"""
        if edge_region.size == 0:
            return True
        
        # 计算边缘变化的标准差
        edge_std = np.std(edge_region)
        return edge_std > 30  # 阈值可调整
    
    def extract_edge_features(self, image_path):
        """提取断口特征"""
        image = cv2.imread(str(image_path))
        if image is None:
            return np.zeros(512)  # 返回零向量
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 轮廓检测
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return np.zeros(512)
        
        # 获取主轮廓
        main_contour = max(contours, key=cv2.contourArea)
        
        # 提取轮廓特征
        features = []
        
        # 轮廓周长
        perimeter = cv2.arcLength(main_contour, True)
        features.append(perimeter)
        
        # 轮廓面积
        area = cv2.contourArea(main_contour)
        features.append(area)
        
        # 长宽比
        x, y, w, h = cv2.boundingRect(main_contour)
        aspect_ratio = w / h if h > 0 else 0
        features.append(aspect_ratio)
        
        # 填充度
        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        features.append(solidity)
        
        # 将特征扩展到512维
        while len(features) < 512:
            features.extend(features[:min(len(features), 512-len(features))])
        
        return np.array(features[:512])
    
    def extract_texture_features(self, image_path):
        """提取纹理特征（竹简纹路）"""
        image = cv2.imread(str(image_path))
        if image is None:
            return np.zeros(256)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # LBP纹理特征
        lbp = self._calculate_lbp(gray)
        hist_lbp = cv2.calcHist([lbp], [0], None, [256], [0, 256])
        hist_lbp = hist_lbp.flatten() / np.sum(hist_lbp)  # 归一化
        
        return hist_lbp
    
    def _calculate_lbp(self, image):
        """计算局部二值模式（LBP）"""
        h, w = image.shape
        lbp = np.zeros((h-2, w-2), dtype=np.uint8)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = image[i, j]
                code = 0
                
                # 8邻域
                neighbors = [
                    image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                    image[i, j+1], image[i+1, j+1], image[i+1, j],
                    image[i+1, j-1], image[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= (1 << k)
                
                lbp[i-1, j-1] = code
        
        return lbp

class BambooSlipDataProcessor:
    """竹简数据处理器 - 优化版"""
    
    def __init__(self, yizhuihe_path, excel_path):
        self.yizhuihe_path = Path(yizhuihe_path)
        self.excel_path = excel_path
        self.image_processor = BambooSlipImageProcessor()
        self.data_pairs = []
        
    def load_excel_data(self):
        """加载Excel中的上下拼sheet数据"""
        try:
            # 只读取"上下拼"sheet，这是主要的训练数据集
            df = pd.read_excel(self.excel_path, sheet_name='上下拼')
            logger.info(f"成功加载上下拼数据，共{len(df)}条记录")
            
            # 打印表头信息用于调试
            logger.info(f"Excel列名: {list(df.columns)}")
            return df
        except Exception as e:
            logger.error(f"加载Excel文件失败: {e}")
            return None
    
    def extract_image_pairs(self, df):
        """
        从Excel数据中提取图片对
        将多个连续拼接的竹简拆分为两两一对的组合
        """
        pairs = []
        
        for _, row in df.iterrows():
            if pd.isna(row.iloc[0]):  # 跳过空行
                continue
                
            group_id = row.iloc[0]  # A列：缀合组号
            
            # 提取B到F列的简号，按照从上到下的顺序
            slip_ids = []
            for i in range(1, 6):  # B到F列
                if i < len(row) and pd.notna(row.iloc[i]):
                    # 处理简号，移除可能的前缀9-
                    slip_id = str(row.iloc[i])
                    if slip_id.startswith('9-'):
                        slip_id = slip_id[2:]
                    slip_ids.append(slip_id)
            
            # 创建相邻简片的配对（两两一对）
            # 这样更好体现断口特征
            for i in range(len(slip_ids) - 1):
                pairs.append({
                    'group_id': group_id,
                    'slip1': slip_ids[i],    # 上面的简
                    'slip2': slip_ids[i + 1], # 下面的简
                    'label': 1,  # 可缀合标记为1
                    'position': 'vertical'   # 上下拼接
                })
        
        logger.info(f"提取到{len(pairs)}个正样本对")
        return pairs
    
    def find_image_files(self, bamboo_id):
        """
        根据竹简ID查找对应的图像文件
        使用正则匹配处理前缀和后缀问题
        """
        import re
        
        # 提取竹简ID的数字部分（如将 "9-282" 转化为 "282"）
        numeric_id = bamboo_id.split('-')[-1]  # 取最后一个分隔段
        
        # 构建正则：以数字部分开头，可接下划线+后缀字符
        pattern = re.compile(f"^{numeric_id}(?:_[a-zA-Z0-9]+)?$")
        
        matches = []
        
        # 在yizhuihe文件夹中搜索
        for folder in self.yizhuihe_path.iterdir():
            if folder.is_dir():
                for img_file in folder.glob('*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        # 去除文件扩展名
                        filename = img_file.stem
                        if pattern.match(filename):
                            matches.append(img_file)
        
        return matches
    
    def get_bamboo_image_pairs(self, joining_pairs):
        """
        根据缀合对获取对应的图像文件对
        """
        image_pairs = []
        
        for pair in joining_pairs:
            img1_files = self.find_image_files(pair['slip1'])
            img2_files = self.find_image_files(pair['slip2'])
            
            if img1_files and img2_files:
                # 选择第一个找到的文件
                image_pairs.append({
                    'img1_path': img1_files[0],
                    'img2_path': img2_files[0],
                    'label': pair['label'],
                    'group_id': pair['group_id'],
                    'slip1_id': pair['slip1'],
                    'slip2_id': pair['slip2']
                })
        
        logger.info(f"成功匹配到{len(image_pairs)}对图像")
        return image_pairs
    
    def create_negative_samples(self, positive_pairs, ratio=1.0):
        """创建负样本 - 随机抽取不可缀合的竹简对"""
        # 获取所有简号
        all_slips = set()
        for pair in positive_pairs:
            all_slips.add(pair['slip1'])
            all_slips.add(pair['slip2'])
        
        all_slips = list(all_slips)
        negative_pairs = []
        
        # 创建不可缀合的随机配对
        num_negative = int(len(positive_pairs) * ratio)
        
        while len(negative_pairs) < num_negative:
            slip1 = random.choice(all_slips)
            slip2 = random.choice(all_slips)
            
            if slip1 != slip2:
                # 检查是否已经是正样本
                is_positive = any(
                    (p['slip1'] == slip1 and p['slip2'] == slip2) or
                    (p['slip1'] == slip2 and p['slip2'] == slip1)
                    for p in positive_pairs
                )
                
                if not is_positive:
                    negative_pairs.append({
                        'group_id': 'negative',
                        'slip1': slip1,
                        'slip2': slip2,
                        'label': 0,  # 不可缀合标记为0
                        'position': 'random'
                    })
        
        logger.info(f"创建了{len(negative_pairs)}个负样本对")
        return negative_pairs
    
    def preprocess_image(self, img_path, target_size=(256, 512)):
        """图像预处理 - 使用优化的预处理方法"""
        return self.image_processor.preprocess_bamboo_image(img_path, target_size)

class BambooSlipDataset(Dataset):
    """竹简数据集 - 支持数据增强"""
    
    def __init__(self, data_pairs, processor, transform=None, augment=False):
        self.data_pairs = data_pairs
        self.processor = processor
        self.transform = transform
        self.augment = augment
        
        # 过滤无效的数据对
        self.valid_pairs = []
        for pair in data_pairs:
            img1_files = processor.find_image_files(pair['slip1'])
            img2_files = processor.find_image_files(pair['slip2'])
            if img1_files and img2_files:
                self.valid_pairs.append(pair)
        
        logger.info(f"数据集包含{len(self.valid_pairs)}个有效样本")
        
        # 数据增强变换
        if augment:
            self.augment_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
                transforms.RandomRotation(degrees=10),   # ±10°随机旋转
                transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 亮度/对比度抖动
                transforms.RandomAffine(degrees=0, shear=10),  # 仿射剪切变换
                transforms.ToTensor()
            ])
        else:
            self.augment_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor()
            ])
        
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        pair = self.valid_pairs[idx]
        
        # 加载图片
        img1_files = self.processor.find_image_files(pair['slip1'])
        img2_files = self.processor.find_image_files(pair['slip2'])
        
        # 选择第一个找到的图片文件
        img1 = self.processor.preprocess_image(img1_files[0])
        img2 = self.processor.preprocess_image(img2_files[0])
        
        if img1 is None or img2 is None:
            # 返回零张量作为占位符
            return torch.zeros(3, 512, 256), torch.zeros(3, 512, 256), torch.tensor(0.0)
        
        # 转换为PIL图像以便进行数据增强
        img1_pil = (img1 * 255).astype(np.uint8)
        img2_pil = (img2 * 255).astype(np.uint8)
        
        # 应用数据增强
        if self.augment:
            # 为了保证配对的图片使用相同的随机变换，我们需要固定随机种子
            seed = np.random.randint(2147483647)
            
            # 对第一张图片应用变换
            np.random.seed(seed)
            torch.manual_seed(seed)
            img1 = self.augment_transform(img1_pil)
            
            # 对第二张图片应用相同的变换
            np.random.seed(seed)
            torch.manual_seed(seed)
            img2 = self.augment_transform(img2_pil)
        else:
            img1 = self.augment_transform(img1_pil)
            img2 = self.augment_transform(img2_pil)
        
        label = torch.tensor(pair['label'], dtype=torch.float32)
        
        return img1, img2, label

class SiameseNetwork(nn.Module):
    """优化的孪生网络 - 基于ResNet50 + 注意力机制"""
    
    def __init__(self, embedding_dim=256):
        super(SiameseNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        
        # 使用预训练的ResNet50作为特征提取器
        import torchvision.models as models
        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # 特征映射层 - ResNet50输出2048维
        self.embedding = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, embedding_dim)
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # 改进的特征融合方式 - 使用拼接+多层感知机
        # 输入维度: embedding_dim * 4 (feat1 + feat2 + diff + mul)
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward_one(self, x):
        """单路前向传播，提取特征"""
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.embedding(x)
        return x
    
    def forward(self, x1, x2):
        """双路前向传播，计算相似度"""
        # 提取两个图像的特征
        feat1 = self.forward_one(x1)
        feat2 = self.forward_one(x2)
        
        # 应用注意力权重
        attn1 = self.attention(feat1)
        attn2 = self.attention(feat2)
        feat1_weighted = feat1 * attn1
        feat2_weighted = feat2 * attn2
        
        # 连接特征向量和它们的差异/乘积
        diff = torch.abs(feat1_weighted - feat2_weighted)  # 特征差异
        mul = feat1_weighted * feat2_weighted              # 特征乘积
        
        # 融合所有特征信息
        fused = torch.cat([feat1_weighted, feat2_weighted, diff, mul], dim=1)
        
        # 计算相似度
        similarity = self.fusion(fused)
        
        return similarity.squeeze(), feat1, feat2

class ContrastiveLoss(nn.Module):
    """对比损失函数"""
    
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output, feat1, feat2, target):
        # 计算欧氏距离
        euclidean_distance = nn.functional.pairwise_distance(feat1, feat2)
        
        # 确保target是浮点类型
        target_float = target.float()
        
        # 对比损失
        loss_contrastive = torch.mean(
            (1 - target_float) * torch.pow(euclidean_distance, 2) +
            target_float * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        
        # 二元交叉熵损失
        loss_bce = nn.functional.binary_cross_entropy(output.view_as(target), target_float)
        
        # 只使用BCE损失（根据您的经验，对比损失可能导致梯度不稳定）
        return loss_bce

class BambooSlipMatcher:
    """竹简缀合匹配器 - 优化版"""
    
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        self.model = SiameseNetwork().to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"模型已从 {model_path} 加载")
    
    def _adjust_batch_size(self, dataloader, min_batch_size=2):
        """动态调整batch_size，避免单个样本批次"""
        original_batch_size = dataloader.batch_size
        if len(dataloader.dataset) == 1:
            # 如果数据集只有一个样本，创建新的DataLoader
            new_dataloader = DataLoader(
                dataloader.dataset, 
                batch_size=min_batch_size,
                shuffle=dataloader.drop_last,
                num_workers=dataloader.num_workers
            )
            return new_dataloader
        return dataloader
    
    def train(self, train_loader, val_loader=None, epochs=10, lr=0.0001, save_path=None):
        """训练模型 - 使用优化的训练策略"""
        
        # 动态调整batch_size
        train_loader = self._adjust_batch_size(train_loader)
        if val_loader:
            val_loader = self._adjust_batch_size(val_loader)
        
        # 损失函数和优化器
        criterion = ContrastiveLoss()
        
        # 使用AdamW优化器，降低学习率避免过拟合
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=1e-4
        )
        
        # 使用ReduceLROnPlateau调度器动态调整学习率
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        best_val_acc = 0.0
        train_losses = []
        train_accuracies = []
        val_accuracies = []
        val_f1_scores = []
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            
            for batch_idx, (img1, img2, labels) in enumerate(progress_bar):
                # 跳过批次大小为1的批次
                if img1.size(0) == 1:
                    continue
                    
                img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                # 前向传播
                outputs, feat1, feat2 = self.model(img1, img2)
                
                # 计算损失
                loss = criterion(outputs, feat1, feat2, labels)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 更新进度条
                current_acc = 100 * correct / total if total > 0 else 0
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })
            
            avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
            train_acc = 100 * correct / total if total > 0 else 0
            train_losses.append(avg_loss)
            train_accuracies.append(train_acc)
            
            logger.info(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%')
            
            # 验证阶段
            if val_loader:
                val_acc, val_f1 = self.evaluate(val_loader)
                val_accuracies.append(val_acc)
                val_f1_scores.append(val_f1)
                logger.info(f'Validation Acc: {val_acc:.2f}%, F1: {val_f1:.4f}')
                
                # 更新学习率调度器
                scheduler.step(avg_loss)
                
                # 保存最佳模型
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    if save_path:
                        self.save_model(save_path)
                        logger.info(f'新的最佳模型已保存，验证准确率: {val_acc:.2f}%')
        
        return train_losses, train_accuracies, val_accuracies, val_f1_scores
    
    def evaluate(self, data_loader):
        """评估模型"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for img1, img2, labels in data_loader:
                if img1.size(0) == 1:  # 跳过批次大小为1的批次
                    continue
                    
                img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)
                
                outputs, _, _ = self.model(img1, img2)
                predicted = (outputs > 0.5).float()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        if len(all_predictions) == 0:
            return 0.0, 0.0
        
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)
        
        return accuracy * 100, f1
    
    def predict(self, img1_path, img2_path, processor):
        """预测两个竹简是否可以缀合"""
        self.model.eval()
        
        # 预处理图像
        img1 = processor.preprocess_image(img1_path)
        img2 = processor.preprocess_image(img2_path)
        
        if img1 is None or img2 is None:
            return 0.0, "图像加载失败"
        
        # 转换为tensor
        img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            similarity, _, _ = self.model(img1, img2)
            confidence = similarity.item()
        
        result = "可以缀合" if confidence > 0.5 else "不可缀合"
        
        return confidence, result
    
    def save_model(self, path):
        """保存模型"""
        torch.save(self.model.state_dict(), path)
        logger.info(f"模型已保存到 {path}")
    
    def load_model(self, path):
        """加载模型"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"模型已从 {path} 加载")

def plot_training_history(train_losses, train_accuracies, val_accuracies, val_f1_scores, save_path=None):
    """绘制训练历史 - 四张分开的折线图"""
    epochs = range(1, len(train_losses) + 1)
    
    # 1. 训练损失图
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, 'b-', linewidth=3, marker='o', markersize=6)
    plt.title('Training Loss', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        base_path = save_path.replace('.png', '')
        loss_path = f"{base_path}_train_loss.png"
        plt.savefig(loss_path, dpi=300, bbox_inches='tight')
        logger.info(f"训练损失图已保存: {loss_path}")
    plt.show()
    
    # 2. 训练准确率图
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_accuracies, 'g-', linewidth=3, marker='s', markersize=6)
    plt.title('Training Accuracy', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        base_path = save_path.replace('.png', '')
        acc_path = f"{base_path}_train_accuracy.png"
        plt.savefig(acc_path, dpi=300, bbox_inches='tight')
        logger.info(f"训练准确率图已保存: {acc_path}")
    plt.show()
    
    # 3. 验证准确率图
    if val_accuracies:
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, val_accuracies, 'r-', linewidth=3, marker='^', markersize=6)
        plt.title('Validation Accuracy', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Accuracy (%)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            base_path = save_path.replace('.png', '')
            val_acc_path = f"{base_path}_val_accuracy.png"
            plt.savefig(val_acc_path, dpi=300, bbox_inches='tight')
            logger.info(f"验证准确率图已保存: {val_acc_path}")
        plt.show()
    
    # 4. F1分数图
    if val_f1_scores:
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, val_f1_scores, 'm-', linewidth=3, marker='d', markersize=6)
        plt.title('F1 Score', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('F1 Score', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            base_path = save_path.replace('.png', '')
            f1_path = f"{base_path}_f1_score.png"
            plt.savefig(f1_path, dpi=300, bbox_inches='tight')
            logger.info(f"F1分数图已保存: {f1_path}")
        plt.show()
    
    logger.info("✅ 四张训练历史图表已生成并保存为独立图片")

def main():
    """主函数 - 优化的训练流程"""
    # 配置路径
    yizhuihe_path = "e:/ai_final/yizhuihe"
    excel_path = "e:/ai_final/zhuihetongji.xlsx"
    model_save_path = "e:/ai_final/results/bamboo_slip_model_optimized.pth"
    
    # 创建结果目录
    os.makedirs("e:/ai_final/results", exist_ok=True)
    
    # 初始化处理器
    processor = BambooSlipDataProcessor(yizhuihe_path, excel_path)
    
    # 加载数据 - 只使用"上下拼"数据集
    logger.info("加载上下拼数据...")
    df = processor.load_excel_data()
    if df is None:
        logger.error("无法加载Excel数据，程序退出")
        return
    
    # 提取图片对 - 两两配对提取断口特征
    logger.info("提取图片对...")
    positive_pairs = processor.extract_image_pairs(df)
    negative_pairs = processor.create_negative_samples(positive_pairs, ratio=1.0)
    
    # 合并数据
    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)
    
    # 8:2 训练测试集划分
    train_pairs, val_pairs = train_test_split(all_pairs, test_size=0.2, random_state=42)
    
    logger.info(f"训练集: {len(train_pairs)} 样本")
    logger.info(f"验证集: {len(val_pairs)} 样本")
    
    # 创建数据集 - 训练集使用数据增强
    train_dataset = BambooSlipDataset(train_pairs, processor, augment=True)
    val_dataset = BambooSlipDataset(val_pairs, processor, augment=False)
    
    # 动态调整batch_size，避免单个样本批次
    batch_size = 10
    if len(train_dataset) < batch_size:
        batch_size = max(2, len(train_dataset))
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        drop_last=True  # 丢弃最后一个不完整的批次
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        drop_last=True
    )
    
    # 初始化模型
    matcher = BambooSlipMatcher()
    
    # 训练模型 - 使用优化的超参数
    logger.info("开始训练模型...")
    train_losses, train_accuracies, val_accuracies, val_f1_scores = matcher.train(
        train_loader, val_loader, 
        epochs=10,  # 设置训练轮数为10
        lr=0.0001,  # 降低学习率避免过拟合
        save_path=model_save_path
    )
    
    # 绘制训练历史
    plot_training_history(train_losses, train_accuracies, val_accuracies, val_f1_scores, "e:/ai_final/results/training_history_optimized.png")
    
    # 最终评估
    logger.info("最终评估...")
    final_acc, final_f1 = matcher.evaluate(val_loader)
    logger.info(f"最终验证准确率: {final_acc:.2f}%")
    logger.info(f"最终F1分数: {final_f1:.4f}")
    
    # 保存训练配置和结果
    results = {
        'final_accuracy': final_acc,
        'final_f1_score': final_f1,
        'train_samples': len(train_pairs),
        'val_samples': len(val_pairs),
        'positive_samples': len(positive_pairs),
        'negative_samples': len(negative_pairs),
        'model_architecture': 'ResNet50 + Attention + Feature Fusion',
        'data_augmentation': True,
        'key_region_extraction': True,
        'edge_enhancement': True
    }
    
    import json
    with open("e:/ai_final/results/training_results_optimized.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info("训练完成！优化版模型已保存。")

if __name__ == "__main__":
    main()
