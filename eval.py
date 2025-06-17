import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import glob
from models import RCF


class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        自定义数据集加载器
        """
        self.image_dir = image_dir
        self.transform = transform
        
        # 支持多种图像格式
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        self.image_files = []
        for ext in extensions:
            self.image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        
        self.image_files.sort()
        print(f"Found {len(self.image_files)} images in {image_dir}")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        filename = os.path.splitext(os.path.basename(img_path))[0]
        return image, filename


class EdgeEvaluator:
    def __init__(self, tolerance=0.0075):
        """
        边缘检测评估器
        
        Args:
            tolerance: 匹配容忍度，默认0.0075
        """
        self.tolerance = tolerance
        
    def compute_boundaries_match(self, pred_edge, gt_edge):
        """
        计算预测边缘与真实边缘的匹配
        """
        # 确保输入格式正确
        if pred_edge.max() > 1:
            pred_edge = pred_edge.astype(np.float32) / 255.0
        if gt_edge.max() > 1:
            gt_edge = gt_edge.astype(np.float32) / 255.0
            
        # 获取真实边缘点坐标
        gt_points = np.where(gt_edge > 0.5)
        if len(gt_points[0]) == 0:
            return np.zeros_like(pred_edge), np.zeros_like(gt_edge)
            
        gt_coords = np.column_stack((gt_points[0], gt_points[1]))
        
        # 获取预测边缘点坐标  
        pred_points = np.where(pred_edge > 0)
        if len(pred_points[0]) == 0:
            return np.zeros_like(pred_edge), np.zeros_like(gt_edge)
            
        pred_coords = np.column_stack((pred_points[0], pred_points[1]))
        
        # 计算距离阈值
        h, w = pred_edge.shape
        max_dist = self.tolerance * np.sqrt(h*h + w*w)
        
        match_pred = np.zeros_like(pred_edge)
        match_gt = np.zeros_like(gt_edge)
        
        # 对每个预测点找最近的真实点
        for pred_y, pred_x in pred_coords:
            distances = np.sqrt((gt_coords[:, 0] - pred_y)**2 + (gt_coords[:, 1] - pred_x)**2)
            min_dist = np.min(distances)
            if min_dist <= max_dist:
                match_pred[pred_y, pred_x] = pred_edge[pred_y, pred_x]
                
        # 对每个真实点找最近的预测点
        for gt_y, gt_x in gt_coords:
            distances = np.sqrt((pred_coords[:, 0] - gt_y)**2 + (pred_coords[:, 1] - gt_x)**2)
            min_dist = np.min(distances)
            if min_dist <= max_dist:
                match_gt[gt_y, gt_x] = 1.0
                
        return match_pred, match_gt
    
    def evaluate_single_image(self, pred_edge, gt_edge, thresholds):
        """
        评估单张图像在不同阈值下的性能
        """
        precision_list = []
        recall_list = []
        f_measure_list = []
        
        for thresh in thresholds:
            # 二值化预测结果
            pred_binary = (pred_edge >= thresh).astype(np.float32)
            
            # 计算匹配
            match_pred, match_gt = self.compute_boundaries_match(pred_binary, gt_edge)
            
            # 计算TP, FP, FN
            tp = np.sum(match_pred > 0)
            fp = np.sum(pred_binary > 0) - tp
            fn = np.sum(gt_edge > 0) - np.sum(match_gt > 0)
            
            # 计算精确率和召回率
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # 计算F-measure
            f_measure = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            precision_list.append(precision)
            recall_list.append(recall)
            f_measure_list.append(f_measure)
            
        return precision_list, recall_list, f_measure_list


def run_inference(model, image_dir, output_dir, device):
    """
    运行RCF模型推理 - 使用与原论文一致的预处理
    """
    print("Starting inference...")
    
    # RCF原论文的预处理参数
    mean = np.array([104.00698793, 116.66876762, 122.67891434], dtype=np.float32)
    
    # 获取图像文件
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
    image_files.sort()
    
    if not image_files:
        raise ValueError(f"No images found in {image_dir}")
    
    print(f"Found {len(image_files)} images")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    processed_files = []
    
    with torch.no_grad():
        for idx, img_path in enumerate(image_files):
            filename = os.path.splitext(os.path.basename(img_path))[0]
            
            # 加载图像 (BGR格式，与OpenCV一致)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"Warning: Cannot load image {img_path}")
                continue
                
            # RCF原论文预处理方式
            img = img.astype(np.float32)
            img -= mean  # 减去均值
            img = img.transpose((2, 0, 1))  # HWC -> CHW
            
            # 准备输入张量
            img_tensor = torch.from_numpy(img).unsqueeze(0)  # 添加batch维度
            img_tensor = img_tensor.to(device)
            
            # 模型推理
            results = model(img_tensor)
            
            # 获取融合结果（最后一个输出）
            fuse_result = results[-1].detach().cpu().numpy()
            fuse_result = np.squeeze(fuse_result)  # 移除batch和channel维度
            
            # 转换为0-255范围并保存（不进行1-操作，保持边缘为白色）
            edge_map = (fuse_result * 255).astype(np.uint8)
            output_path = os.path.join(output_dir, f"{filename}.png")
            cv2.imwrite(output_path, edge_map)
            
            processed_files.append(filename)
            print(f"Processed {idx+1}/{len(image_files)}: {filename}")
    
    print(f"Inference completed. Results saved to {output_dir}")
    return processed_files


def evaluate_results(pred_dir, gt_dir, evaluator):
    """
    评估预测结果
    """
    print("Starting evaluation...")
    
    # 获取预测文件列表
    pred_files = glob.glob(os.path.join(pred_dir, "*.png"))
    pred_files.sort()
    
    if not pred_files:
        raise ValueError(f"No prediction files found in {pred_dir}")
    
    # 阈值设置
    thresholds = np.linspace(0.01, 0.99, 99)
    
    # 存储所有图像的结果
    all_precisions = []
    all_recalls = []
    all_f_measures = []
    image_best_f = []
    
    processed_count = 0
    
    for pred_file in pred_files:
        base_name = os.path.splitext(os.path.basename(pred_file))[0]
        
        # 查找对应的真实标签
        gt_file = os.path.join(gt_dir, f"{base_name}.png")
        if not os.path.exists(gt_file):
            gt_file = os.path.join(gt_dir, f"{base_name}.jpg")
        if not os.path.exists(gt_file):
            print(f"Warning: Ground truth not found for {base_name}")
            continue
        
        # 加载预测结果和真实标签
        pred_edge = cv2.imread(pred_file, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        gt_edge = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        
        # 评估单张图像
        precisions, recalls, f_measures = evaluator.evaluate_single_image(pred_edge, gt_edge, thresholds)
        
        all_precisions.append(precisions)
        all_recalls.append(recalls)
        all_f_measures.append(f_measures)
        
        # 记录该图像的最佳F-measure (OIS)
        image_best_f.append(max(f_measures))
        processed_count += 1
        
        print(f"Evaluated {processed_count}: {base_name}")
    
    if processed_count == 0:
        raise ValueError("No matching prediction-ground truth pairs found")
    
    # 转换为numpy数组
    all_precisions = np.array(all_precisions)
    all_recalls = np.array(all_recalls)
    all_f_measures = np.array(all_f_measures)
    
    # 计算ODS: 在所有图像上平均后找到最优阈值
    mean_f_measures = np.mean(all_f_measures, axis=0)
    ods_idx = np.argmax(mean_f_measures)
    ods_f = mean_f_measures[ods_idx]
    ods_threshold = thresholds[ods_idx]
    
    # 计算OIS: 每张图像最佳F-measure的平均
    ois_f = np.mean(image_best_f)
    
    # 计算AP (Average Precision)
    mean_precisions = np.mean(all_precisions, axis=0)
    mean_recalls = np.mean(all_recalls, axis=0)
    
    # 按召回率排序计算AP
    sorted_indices = np.argsort(mean_recalls)
    sorted_recalls = mean_recalls[sorted_indices]
    sorted_precisions = mean_precisions[sorted_indices]
    
    # 使用新的trapezoid函数替代已弃用的trapz
    try:
        ap = np.trapezoid(sorted_precisions, sorted_recalls)
    except AttributeError:
        # 如果是旧版本NumPy，回退到trapz
        ap = np.trapz(sorted_precisions, sorted_recalls)
    
    print(f"\n=== RCF Evaluation Results ===")
    print(f"Processed {processed_count} image pairs")
    print(f"ODS F-measure: {ods_f:.4f} (threshold: {ods_threshold:.3f})")
    print(f"OIS F-measure: {ois_f:.4f}")
    print(f"Average Precision (AP): {ap:.4f}")
    
    return {
        'ods_f': ods_f,
        'ois_f': ois_f,
        'ap': ap,
        'ods_threshold': ods_threshold,
        'mean_precisions': mean_precisions,
        'mean_recalls': mean_recalls,
        'thresholds': thresholds
    }


def plot_pr_curve(results, save_path):
    """
    绘制PR曲线
    """
    plt.figure(figsize=(10, 8))
    
    # 绘制PR曲线
    plt.plot(results['mean_recalls'], results['mean_precisions'], 
             'b-', linewidth=2, label=f"RCF (AP={results['ap']:.3f})")
    
    # 标记ODS点
    ods_idx = np.argmax(results['mean_precisions'] * results['mean_recalls'] / 
                       (results['mean_precisions'] + results['mean_recalls'] + 1e-8))
    plt.plot(results['mean_recalls'][ods_idx], results['mean_precisions'][ods_idx], 
             'ro', markersize=8, label=f"ODS F={results['ods_f']:.3f}")
    
    # 绘制等F-measure线
    for f_val in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        recall_range = np.linspace(0.01, 1.0, 100)
        precision_iso = f_val * recall_range / (2 * recall_range - f_val)
        precision_iso = np.clip(precision_iso, 0, 1)
        valid_idx = (precision_iso >= 0) & (precision_iso <= 1) & (recall_range >= f_val/2)
        if np.any(valid_idx):
            plt.plot(recall_range[valid_idx], precision_iso[valid_idx], 
                    'g--', alpha=0.3, linewidth=1)
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve for Edge Detection', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axis([0, 1, 0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"PR curve saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='RCF Edge Detection Evaluation')
    parser.add_argument('--checkpoint', default='checkpoint_epoch7.pth', 
                       help='Path to model checkpoint')
    parser.add_argument('--image-dir', default=r'C:\Users\25068\Desktop\RCF_pytorch\eval\0.0_1_0',
                       help='Directory containing input images')
    parser.add_argument('--output-dir', default=r'C:\Users\25068\Desktop\RCF_pytorch\eval\0.0_1_0_res',
                       help='Directory to save prediction results')
    parser.add_argument('--gt-dir', default=r'C:\Users\25068\Desktop\RCF_pytorch\eval\0.0_1_0_gt',
                       help='Directory containing ground truth images')
    parser.add_argument('--gpu', default='0', help='GPU ID')
    parser.add_argument('--tolerance', default=0.0075, type=float,
                       help='Tolerance for edge matching')
    
    args = parser.parse_args()
    
    # 设置GPU
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    print("Loading RCF model...")
    model = RCF().to(device)
    
    if os.path.isfile(args.checkpoint):
        print(f"Loading checkpoint from '{args.checkpoint}'")
        try:
            # 尝试使用新的安全模式加载
            checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        except Exception as e:
            print(f"Safe loading failed: {e}")
            print("Falling back to legacy loading mode...")
            # 回退到旧的加载方式
            checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        
        # 处理不同的checkpoint格式
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
        else:
            # 如果checkpoint直接是state_dict
            model.load_state_dict(checkpoint)
        
        print("Checkpoint loaded successfully")
    else:
        raise FileNotFoundError(f"No checkpoint found at '{args.checkpoint}'")
    
    # 运行推理
    processed_files = run_inference(model, args.image_dir, args.output_dir, device)
    
    # 创建评估器
    evaluator = EdgeEvaluator(tolerance=args.tolerance)
    
    # 评估结果
    results = evaluate_results(args.output_dir, args.gt_dir, evaluator)
    
    # 绘制PR曲线
    pr_curve_path = os.path.join(args.output_dir, 'pr_curve.png')
    plot_pr_curve(results, pr_curve_path)
    
    # 保存评估结果
    results_file = os.path.join(args.output_dir, 'evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write("RCF Edge Detection Evaluation Results\n")
        f.write("="*40 + "\n")
        f.write(f"ODS F-measure: {results['ods_f']:.4f}\n")
        f.write(f"OIS F-measure: {results['ois_f']:.4f}\n")
        f.write(f"Average Precision: {results['ap']:.4f}\n")
        f.write(f"ODS Threshold: {results['ods_threshold']:.4f}\n")
        f.write(f"Processed Images: {len(processed_files)}\n")
    
    print(f"\nEvaluation completed successfully!")
    print(f"Results saved to: {results_file}")
    print(f"PR curve saved to: {pr_curve_path}")


if __name__ == '__main__':
    main()