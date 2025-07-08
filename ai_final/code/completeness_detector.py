"""
竹简完整性检测模块
用于判断竹简是否为完整简或残简
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

class CompletenessDetector:
    """竹简完整性检测器"""
    
    def __init__(self):
        self.edge_threshold_low = 50
        self.edge_threshold_high = 150
        self.min_contour_area = 500
        
    def analyze_slip_completeness(self, image_path, visualize=False):
        """
        分析竹简完整性
        返回详细的分析结果
        """
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        # 获取图像尺寸
        h, w = image.shape[:2]
        
        # 转为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 图像预处理
        preprocessed = self._preprocess_image(gray)
        
        # 边缘检测
        edges = cv2.Canny(preprocessed, self.edge_threshold_low, self.edge_threshold_high)
        
        # 轮廓检测
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {
                'is_complete': False,
                'confidence': 0.0,
                'analysis': '无法检测到竹简轮廓'
            }
        
        # 找到最大轮廓（竹简主体）
        main_contour = max(contours, key=cv2.contourArea)
        
        # 基本几何特征
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)
        x, y, width, height = cv2.boundingRect(main_contour)
        
        # 分析结果
        result = {
            'is_complete': False,
            'confidence': 0.0,
            'width': width,
            'height': height,
            'area': area,
            'perimeter': perimeter,
            'aspect_ratio': height / width if width > 0 else 0,
            'break_analysis': {},
            'shape_analysis': {},
            'recommendations': []
        }
        
        # 形状分析
        shape_score = self._analyze_shape_completeness(main_contour, x, y, width, height)
        result['shape_analysis'] = shape_score
        
        # 边缘分析
        edge_analysis = self._analyze_edges(gray, main_contour, x, y, width, height)
        result['break_analysis'] = edge_analysis
        
        # 综合判断
        completeness_score = self._calculate_completeness_score(shape_score, edge_analysis)
        result['confidence'] = completeness_score
        result['is_complete'] = completeness_score > 0.7
        
        # 生成建议
        result['recommendations'] = self._generate_recommendations(result)
        
        # 可视化
        if visualize:
            self._visualize_analysis(image, main_contour, result, image_path)
        
        return result
    
    def _preprocess_image(self, gray_image):
        """图像预处理"""
        # 高斯模糊去噪
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
        # 直方图均衡化增强对比度
        equalized = cv2.equalizeHist(blurred)
        
        return equalized
    
    def _analyze_shape_completeness(self, contour, x, y, width, height):
        """分析形状完整性"""
        # 计算凸包
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        # 填充度（solidity）
        solidity = cv2.contourArea(contour) / hull_area if hull_area > 0 else 0
        
        # 长宽比分析（完整竹简通常细长）
        aspect_ratio = height / width if width > 0 else 0
        aspect_score = 1.0 if 3 < aspect_ratio < 20 else max(0, 1 - abs(aspect_ratio - 8) / 8)
        
        # 轮廓平滑度
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        smoothness = 1.0 / (len(approx) / 10 + 1)  # 顶点越少越平滑
        
        return {
            'solidity': solidity,
            'aspect_ratio': aspect_ratio,
            'aspect_score': aspect_score,
            'smoothness': smoothness,
            'hull_area': hull_area,
            'contour_area': cv2.contourArea(contour)
        }
    
    def _analyze_edges(self, gray_image, contour, x, y, width, height):
        """分析边缘特征"""
        # 提取四个边缘区域
        margin = 10
        
        # 上边缘
        top_region = gray_image[max(0, y-margin):y+margin, x:x+width]
        top_score = self._analyze_edge_regularity(top_region, 'horizontal')
        
        # 下边缘
        bottom_region = gray_image[y+height-margin:min(gray_image.shape[0], y+height+margin), x:x+width]
        bottom_score = self._analyze_edge_regularity(bottom_region, 'horizontal')
        
        # 左边缘
        left_region = gray_image[y:y+height, max(0, x-margin):x+margin]
        left_score = self._analyze_edge_regularity(left_region, 'vertical')
        
        # 右边缘
        right_region = gray_image[y:y+height, x+width-margin:min(gray_image.shape[1], x+width+margin)]
        right_score = self._analyze_edge_regularity(right_region, 'vertical')
        
        return {
            'top': top_score,
            'bottom': bottom_score,
            'left': left_score,
            'right': right_score,
            'overall_edge_score': np.mean([top_score['regularity'], bottom_score['regularity'], 
                                         left_score['regularity'], right_score['regularity']])
        }
    
    def _analyze_edge_regularity(self, edge_region, direction):
        """分析边缘规律性"""
        if edge_region.size == 0:
            return {'regularity': 0.0, 'variance': float('inf'), 'breaks_detected': True}
        
        # 计算边缘变化
        if direction == 'horizontal':
            # 水平方向：分析每列的边缘变化
            edge_profile = np.std(edge_region, axis=0)
        else:
            # 垂直方向：分析每行的边缘变化
            edge_profile = np.std(edge_region, axis=1)
        
        # 计算规律性指标
        variance = np.var(edge_profile)
        mean_std = np.mean(edge_profile)
        
        # 检测突然的变化（可能的断口）
        if len(edge_profile) > 1:
            diff = np.diff(edge_profile)
            sudden_changes = np.sum(np.abs(diff) > np.std(diff) * 2)
            breaks_detected = sudden_changes > len(edge_profile) * 0.1
        else:
            breaks_detected = False
        
        # 规律性评分（方差越小越规律）
        regularity = 1.0 / (1.0 + variance / 100)
        
        return {
            'regularity': regularity,
            'variance': variance,
            'mean_std': mean_std,
            'breaks_detected': breaks_detected,
            'sudden_changes': sudden_changes if 'sudden_changes' in locals() else 0
        }
    
    def _calculate_completeness_score(self, shape_analysis, edge_analysis):
        """计算完整性评分"""
        # 形状评分权重
        shape_weight = 0.4
        edge_weight = 0.6
        
        # 形状评分
        shape_score = (
            shape_analysis['solidity'] * 0.3 +
            shape_analysis['aspect_score'] * 0.4 +
            shape_analysis['smoothness'] * 0.3
        )
        
        # 边缘评分
        edge_score = edge_analysis['overall_edge_score']
        
        # 综合评分
        total_score = shape_score * shape_weight + edge_score * edge_weight
        
        return min(1.0, max(0.0, total_score))
    
    def _generate_recommendations(self, analysis_result):
        """生成分析建议"""
        recommendations = []
        
        if analysis_result['confidence'] < 0.3:
            recommendations.append("图像质量较差，建议重新拍摄或预处理")
        
        if analysis_result['break_analysis']['top']['breaks_detected']:
            recommendations.append("检测到上端可能存在断口")
        
        if analysis_result['break_analysis']['bottom']['breaks_detected']:
            recommendations.append("检测到下端可能存在断口")
        
        if analysis_result['break_analysis']['left']['breaks_detected']:
            recommendations.append("检测到左侧可能存在断口")
        
        if analysis_result['break_analysis']['right']['breaks_detected']:
            recommendations.append("检测到右侧可能存在断口")
        
        if analysis_result['shape_analysis']['aspect_ratio'] < 2:
            recommendations.append("长宽比异常，可能为残片或横向放置")
        
        if analysis_result['shape_analysis']['solidity'] < 0.8:
            recommendations.append("形状不规则，可能存在缺失部分")
        
        if not recommendations:
            if analysis_result['is_complete']:
                recommendations.append("竹简形状完整，边缘规律")
            else:
                recommendations.append("需要进一步人工确认")
        
        return recommendations
    
    def _visualize_analysis(self, original_image, contour, analysis_result, image_path):
        """可视化分析结果"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'竹简完整性分析 - {image_path.name}', fontsize=16)
        
        # 原图和轮廓
        axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        
        # 绘制轮廓
        contour_image = original_image.copy()
        cv2.drawContours(contour_image, [contour], -1, (0, 255, 0), 2)
        axes[0, 0].imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('原图与检测轮廓')
        axes[0, 0].axis('off')
        
        # 边缘检测结果
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.edge_threshold_low, self.edge_threshold_high)
        axes[0, 1].imshow(edges, cmap='gray')
        axes[0, 1].set_title('边缘检测')
        axes[0, 1].axis('off')
        
        # 分析结果文本
        axes[1, 0].text(0.1, 0.9, f"完整性: {'是' if analysis_result['is_complete'] else '否'}", 
                       fontsize=12, transform=axes[1, 0].transAxes)
        axes[1, 0].text(0.1, 0.8, f"置信度: {analysis_result['confidence']:.3f}", 
                       fontsize=12, transform=axes[1, 0].transAxes)
        axes[1, 0].text(0.1, 0.7, f"长宽比: {analysis_result['aspect_ratio']:.2f}", 
                       fontsize=12, transform=axes[1, 0].transAxes)
        axes[1, 0].text(0.1, 0.6, f"填充度: {analysis_result['shape_analysis']['solidity']:.3f}", 
                       fontsize=12, transform=axes[1, 0].transAxes)
        
        # 建议
        recommendation_text = '\n'.join(analysis_result['recommendations'][:5])  # 最多显示5条
        axes[1, 0].text(0.1, 0.4, f"建议:\n{recommendation_text}", 
                       fontsize=10, transform=axes[1, 0].transAxes, 
                       verticalalignment='top')
        axes[1, 0].set_title('分析结果')
        axes[1, 0].axis('off')
        
        # 边缘分析柱状图
        edge_scores = [
            analysis_result['break_analysis']['top']['regularity'],
            analysis_result['break_analysis']['right']['regularity'],
            analysis_result['break_analysis']['bottom']['regularity'],
            analysis_result['break_analysis']['left']['regularity']
        ]
        edge_labels = ['上', '右', '下', '左']
        
        bars = axes[1, 1].bar(edge_labels, edge_scores)
        axes[1, 1].set_title('各边缘规律性评分')
        axes[1, 1].set_ylabel('规律性评分')
        axes[1, 1].set_ylim(0, 1)
        
        # 为柱状图添加颜色
        for i, (bar, score) in enumerate(zip(bars, edge_scores)):
            if score > 0.7:
                bar.set_color('green')
            elif score > 0.4:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        plt.tight_layout()
        plt.show()
    
    def batch_analyze(self, image_paths, save_results=True, results_path=None):
        """批量分析竹简完整性"""
        results = []
        
        for image_path in image_paths:
            logger.info(f"分析: {image_path}")
            result = self.analyze_slip_completeness(image_path)
            
            if result:
                result['image_path'] = str(image_path)
                results.append(result)
        
        # 统计结果
        if results:
            complete_count = sum(1 for r in results if r['is_complete'])
            incomplete_count = len(results) - complete_count
            
            logger.info(f"分析完成: 共{len(results)}个样本")
            logger.info(f"完整简: {complete_count}个 ({complete_count/len(results)*100:.1f}%)")
            logger.info(f"残简: {incomplete_count}个 ({incomplete_count/len(results)*100:.1f}%)")
            
            # 保存结果
            if save_results and results_path:
                import json
                with open(results_path, 'w', encoding='utf-8') as f:
                    # 转换numpy类型为Python原生类型以便JSON序列化
                    json_results = []
                    for result in results:
                        json_result = {}
                        for key, value in result.items():
                            if isinstance(value, np.ndarray):
                                json_result[key] = value.tolist()
                            elif isinstance(value, np.integer):
                                json_result[key] = int(value)
                            elif isinstance(value, np.floating):
                                json_result[key] = float(value)
                            else:
                                json_result[key] = value
                        json_results.append(json_result)
                    
                    json.dump(json_results, f, ensure_ascii=False, indent=2)
                logger.info(f"结果已保存到: {results_path}")
        
        return results

def demonstrate_completeness_detection():
    """演示完整性检测功能"""
    from pathlib import Path
    
    # 初始化检测器
    detector = CompletenessDetector()
    
    # 测试路径
    test_path = Path("e:/ai_final/weizhuihe")
    
    if test_path.exists():
        # 获取一些测试图片
        image_files = list(test_path.glob("*.jpg"))[:5]  # 测试前5张图片
        
        if image_files:
            logger.info(f"开始测试完整性检测，共{len(image_files)}张图片")
            
            # 批量分析
            results = detector.batch_analyze(
                image_files, 
                save_results=True, 
                results_path="e:/ai_final/results/completeness_analysis.json"
            )
            
            # 显示详细结果
            for result in results[:3]:  # 显示前3个结果
                print(f"\n图片: {Path(result['image_path']).name}")
                print(f"完整性: {'是' if result['is_complete'] else '否'}")
                print(f"置信度: {result['confidence']:.3f}")
                print(f"建议: {'; '.join(result['recommendations'])}")
        else:
            logger.warning("未找到测试图片")
    else:
        logger.warning(f"测试路径不存在: {test_path}")

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行演示
    demonstrate_completeness_detection()
