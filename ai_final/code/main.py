"""
竹简缀合识别系统 - 主运行脚本
整合完整性检测和缀合匹配功能
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import json
import random

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bamboo_slip_matcher import (
    BambooSlipDataProcessor, BambooSlipDataset, 
    SiameseNetwork, BambooSlipMatcher, 
    plot_training_history
)
from completeness_detector import CompletenessDetector

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bamboo_slip_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BambooSlipAnalysisSystem:
    """竹简分析系统主类"""
    
    def __init__(self, config):
        self.config = config
        self.completeness_detector = CompletenessDetector()
        self.data_processor = None
        self.matcher = None
        
        # 创建必要的目录
        os.makedirs(self.config['results_dir'], exist_ok=True)
        os.makedirs(self.config['models_dir'], exist_ok=True)
    
    def analyze_completeness(self, image_dir, output_file=None):
        """分析竹简完整性"""
        logger.info("开始分析竹简完整性...")
        
        image_dir = Path(image_dir)
        image_files = []
        
        # 支持多种图像格式
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(image_dir.glob(ext))
        
        if not image_files:
            logger.warning(f"在目录 {image_dir} 中未找到图像文件")
            return []
        
        # 批量分析
        results = self.completeness_detector.batch_analyze(
            image_files,
            save_results=True,
            results_path=output_file or os.path.join(self.config['results_dir'], 'completeness_results.json')
        )
        
        # 分类结果
        complete_slips = [r for r in results if r['is_complete']]
        incomplete_slips = [r for r in results if not r['is_complete']]
        
        logger.info(f"完整性分析完成:")
        logger.info(f"  - 完整简: {len(complete_slips)}个")
        logger.info(f"  - 残简: {len(incomplete_slips)}个")
        
        return {
            'all_results': results,
            'complete_slips': complete_slips,
            'incomplete_slips': incomplete_slips
        }
    
    def prepare_training_data(self):
        """准备训练数据"""
        logger.info("准备训练数据...")
        
        # 初始化数据处理器
        self.data_processor = BambooSlipDataProcessor(
            self.config['yizhuihe_path'],
            self.config['excel_path']
        )
        
        # 加载Excel数据
        df = self.data_processor.load_excel_data()
        if df is None:
            raise ValueError("无法加载Excel数据")
        
        # 提取图片对
        positive_pairs = self.data_processor.extract_image_pairs(df)
        negative_pairs = self.data_processor.create_negative_samples(positive_pairs, ratio=1.0)
        
        # 合并和打乱数据
        all_pairs = positive_pairs + negative_pairs
        random.shuffle(all_pairs)
        
        logger.info(f"数据准备完成:")
        logger.info(f"  - 正样本: {len(positive_pairs)}对")
        logger.info(f"  - 负样本: {len(negative_pairs)}对")
        logger.info(f"  - 总计: {len(all_pairs)}对")
        
        return all_pairs
    
    def train_matching_model(self, data_pairs=None, epochs=10, batch_size=10, lr=0.0001):
        """训练缀合匹配模型"""
        logger.info("开始训练缀合匹配模型...")
        
        if data_pairs is None:
            data_pairs = self.prepare_training_data()
        
        # 划分训练集和验证集
        from sklearn.model_selection import train_test_split
        train_pairs, val_pairs = train_test_split(data_pairs, test_size=0.2, random_state=42)
        
        # 创建数据集 - 训练集使用数据增强
        train_dataset = BambooSlipDataset(train_pairs, self.data_processor, augment=True)
        val_dataset = BambooSlipDataset(val_pairs, self.data_processor, augment=False)
        
        # 动态调整batch_size，避免单个样本批次
        if len(train_dataset) < batch_size:
            batch_size = max(2, len(train_dataset))
        
        # 创建数据加载器
        from torch.utils.data import DataLoader
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
        self.matcher = BambooSlipMatcher()
        
        # 训练
        model_save_path = os.path.join(self.config['results_dir'], 'bamboo_slip_model_optimized.pth')
        train_losses, train_accuracies, val_accuracies, val_f1_scores = self.matcher.train(
            train_loader, val_loader,
            epochs=epochs, lr=lr,
            save_path=model_save_path
        )
        
        # 绘制训练历史 - 四张独立图片
        plot_path = os.path.join(self.config['results_dir'], 'training_history_optimized.png')
        plot_training_history(train_losses, train_accuracies, val_accuracies, val_f1_scores, plot_path)
        
        # 最终评估
        final_acc, final_f1 = self.matcher.evaluate(val_loader)
        
        # 保存训练结果
        training_results = {
            'final_accuracy': final_acc,
            'final_f1_score': final_f1,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'val_f1_scores': val_f1_scores,
            'train_samples': len(train_pairs),
            'val_samples': len(val_pairs),
            'model_architecture': 'ResNet50 + Attention + Feature Fusion',
            'data_augmentation': True,
            'key_region_extraction': True,
            'edge_enhancement': True
        }
        
        results_path = os.path.join(self.config['results_dir'], 'training_results_optimized.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(training_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"模型训练完成:")
        logger.info(f"  - 最终准确率: {final_acc:.2f}%")
        logger.info(f"  - 最终F1分数: {final_f1:.4f}")
        logger.info(f"  - 模型保存至: {model_save_path}")
        
        return training_results
    
    def predict_matching(self, img1_path, img2_path, model_path=None):
        """预测两个竹简是否可以缀合"""
        if self.matcher is None:
            model_path = model_path or os.path.join(self.config['models_dir'], 'bamboo_slip_matcher.pth')
            self.matcher = BambooSlipMatcher(model_path)
        
        if self.data_processor is None:
            self.data_processor = BambooSlipDataProcessor(
                self.config['yizhuihe_path'],
                self.config['excel_path']
            )
        
        confidence, result = self.matcher.predict(img1_path, img2_path, self.data_processor)
        
        logger.info(f"缀合预测结果:")
        logger.info(f"  - 图片1: {Path(img1_path).name}")
        logger.info(f"  - 图片2: {Path(img2_path).name}")
        logger.info(f"  - 置信度: {confidence:.4f}")
        logger.info(f"  - 结果: {result}")
        
        return confidence, result
    
    def batch_matching_analysis(self, image_dir, output_file=None):
        """批量分析图片的缀合可能性"""
        logger.info("开始批量缀合分析...")
        
        image_dir = Path(image_dir)
        image_files = []
        
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(image_dir.glob(ext))
        
        if len(image_files) < 2:
            logger.warning("需要至少2张图片进行缀合分析")
            return []
        
        # 确保模型已加载
        if self.matcher is None:
            model_path = os.path.join(self.config['models_dir'], 'bamboo_slip_matcher.pth')
            if not os.path.exists(model_path):
                logger.error(f"模型文件不存在: {model_path}")
                logger.error("请先训练模型或提供正确的模型路径")
                return []
            self.matcher = BambooSlipMatcher(model_path)
        
        if self.data_processor is None:
            self.data_processor = BambooSlipDataProcessor(
                self.config['yizhuihe_path'],
                self.config['excel_path']
            )
        
        # 两两比较所有图片
        results = []
        total_pairs = len(image_files) * (len(image_files) - 1) // 2
        
        logger.info(f"将分析 {total_pairs} 个图片对...")
        
        for i in range(len(image_files)):
            for j in range(i + 1, len(image_files)):
                img1_path = image_files[i]
                img2_path = image_files[j]
                
                try:
                    confidence, result = self.matcher.predict(img1_path, img2_path, self.data_processor)
                    
                    results.append({
                        'image1': img1_path.name,
                        'image2': img2_path.name,
                        'confidence': confidence,
                        'prediction': result,
                        'can_match': confidence > 0.5
                    })
                    
                except Exception as e:
                    logger.warning(f"分析失败 {img1_path.name} - {img2_path.name}: {e}")
        
        # 按置信度排序
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 统计结果
        matching_pairs = [r for r in results if r['can_match']]
        
        logger.info(f"批量分析完成:")
        logger.info(f"  - 总配对数: {len(results)}")
        logger.info(f"  - 可缀合配对: {len(matching_pairs)}")
        logger.info(f"  - 平均置信度: {np.mean([r['confidence'] for r in results]):.4f}")
        
        # 保存结果
        if output_file is None:
            output_file = os.path.join(self.config['results_dir'], 'batch_matching_results.json')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结果已保存到: {output_file}")
        
        return results
    
    def comprehensive_analysis(self, weizhuihe_dir, sample_size=50):
        """对未缀合竹简进行综合分析"""
        logger.info("开始综合分析...")
        
        # 1. 完整性分析
        logger.info("步骤1: 完整性分析")
        completeness_results = self.analyze_completeness(
            weizhuihe_dir,
            os.path.join(self.config['results_dir'], 'weizhuihe_completeness.json')
        )
        
        # 2. 从残简中随机选择样本进行缀合分析
        incomplete_slips = completeness_results['incomplete_slips']
        
        if len(incomplete_slips) < 2:
            logger.warning("残简数量不足，无法进行缀合分析")
            return completeness_results
        
        # 随机选择样本
        sample_size = min(sample_size, len(incomplete_slips))
        sample_slips = random.sample(incomplete_slips, sample_size)
        
        logger.info(f"步骤2: 从{len(incomplete_slips)}个残简中选择{len(sample_slips)}个进行缀合分析")
        
        # 提取图片路径
        sample_image_paths = [Path(slip['image_path']) for slip in sample_slips]
        
        # 3. 缀合分析
        matching_results = self.batch_matching_analysis(
            sample_image_paths[0].parent,  # 使用第一个图片的目录
            os.path.join(self.config['results_dir'], 'sample_matching_results.json')
        )
        
        # 4. 生成综合报告
        report = self.generate_analysis_report(completeness_results, matching_results)
        
        report_path = os.path.join(self.config['results_dir'], 'comprehensive_analysis_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"综合分析完成，报告已保存到: {report_path}")
        
        return report
    
    def generate_analysis_report(self, completeness_results, matching_results):
        """生成分析报告"""
        import numpy as np
        
        # 完整性统计
        total_slips = len(completeness_results['all_results'])
        complete_count = len(completeness_results['complete_slips'])
        incomplete_count = len(completeness_results['incomplete_slips'])
        
        # 缀合统计
        total_pairs = len(matching_results)
        matching_pairs = [r for r in matching_results if r['can_match']]
        matching_count = len(matching_pairs)
        
        report = {
            'analysis_summary': {
                'total_bamboo_slips': total_slips,
                'complete_slips': complete_count,
                'incomplete_slips': incomplete_count,
                'completeness_rate': complete_count / total_slips if total_slips > 0 else 0,
                'total_pairs_analyzed': total_pairs,
                'potential_matches': matching_count,
                'matching_rate': matching_count / total_pairs if total_pairs > 0 else 0
            },
            'completeness_distribution': {
                'high_confidence_complete': len([r for r in completeness_results['complete_slips'] if r['confidence'] > 0.8]),
                'medium_confidence_complete': len([r for r in completeness_results['complete_slips'] if 0.6 < r['confidence'] <= 0.8]),
                'low_confidence_complete': len([r for r in completeness_results['complete_slips'] if r['confidence'] <= 0.6]),
                'high_confidence_incomplete': len([r for r in completeness_results['incomplete_slips'] if r['confidence'] > 0.8]),
                'medium_confidence_incomplete': len([r for r in completeness_results['incomplete_slips'] if 0.6 < r['confidence'] <= 0.8]),
                'low_confidence_incomplete': len([r for r in completeness_results['incomplete_slips'] if r['confidence'] <= 0.6])
            },
            'matching_statistics': {
                'average_confidence': np.mean([r['confidence'] for r in matching_results]) if matching_results else 0,
                'high_confidence_matches': len([r for r in matching_pairs if r['confidence'] > 0.8]),
                'medium_confidence_matches': len([r for r in matching_pairs if 0.6 < r['confidence'] <= 0.8]),
                'low_confidence_matches': len([r for r in matching_pairs if 0.5 < r['confidence'] <= 0.6])
            },
            'recommendations': []
        }
        
        # 生成建议
        if report['analysis_summary']['completeness_rate'] < 0.5:
            report['recommendations'].append("大部分竹简为残片，建议重点关注缀合工作")
        
        if report['analysis_summary']['matching_rate'] > 0.1:
            report['recommendations'].append("发现较多潜在缀合对，建议人工验证")
        
        if report['matching_statistics']['high_confidence_matches'] > 0:
            report['recommendations'].append(f"发现{report['matching_statistics']['high_confidence_matches']}个高置信度缀合对，优先处理")
        
        return report

def create_default_config():
    """创建默认配置"""
    return {
        'yizhuihe_path': 'e:/ai_final/yizhuihe',
        'weizhuihe_path': 'e:/ai_final/weizhuihe',
        'excel_path': 'e:/ai_final/zhuihetongji.xlsx',
        'results_dir': 'e:/ai_final/results',
        'models_dir': 'e:/ai_final/models'
    }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='竹简缀合识别系统')
    parser.add_argument('--mode', choices=['train', 'analyze', 'predict', 'comprehensive'], 
                       default='train', help='运行模式')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--img1', type=str, help='预测模式：第一张图片路径')
    parser.add_argument('--img2', type=str, help='预测模式：第二张图片路径')
    parser.add_argument('--image_dir', type=str, help='图片目录路径')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=10, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--sample_size', type=int, default=50, help='综合分析的样本大小')
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = create_default_config()
    
    # 创建分析系统
    system = BambooSlipAnalysisSystem(config)
    
    try:
        if args.mode == 'train':
            logger.info("开始训练模式...")
            system.train_matching_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
            
        elif args.mode == 'analyze':
            image_dir = args.image_dir or config['weizhuihe_path']
            logger.info(f"开始分析模式，分析目录: {image_dir}")
            system.analyze_completeness(image_dir)
            
        elif args.mode == 'predict':
            if not args.img1 or not args.img2:
                logger.error("预测模式需要提供两张图片路径 (--img1 和 --img2)")
                return
            logger.info("开始预测模式...")
            system.predict_matching(args.img1, args.img2)
            
        elif args.mode == 'comprehensive':
            weizhuihe_dir = args.image_dir or config['weizhuihe_path']
            logger.info(f"开始综合分析模式，分析目录: {weizhuihe_dir}")
            system.comprehensive_analysis(weizhuihe_dir, args.sample_size)
            
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        raise

if __name__ == "__main__":
    # 导入numpy，避免在函数中导入
    import numpy as np
    main()
