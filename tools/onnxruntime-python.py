import argparse
import numpy as np
import onnxruntime as ort
print(ort.__file__)
import cv2
import os
import time
from tqdm import tqdm

# 常用的 CLI 启动案例:
#
# 1. 单张图像推理:
# python onnxruntime-python.py --model /path/to/your/model.onnx --image /path/to/your/image.jpg --labels /path/to/your/labels.txt --topk 3
#
# 2. 批量图像推理:
# python onnxruntime-python.py --model /path/to/your/model.onnx --dir /path/to/your/image/directory --labels /path/to/your/labels.txt --threshold 0.6
#
# 3. 批量推理并进行评估:
# python onnxruntime-python.py --model /path/to/your/model.onnx --dir /path/to/your/image/directory --labels /path/to/your/labels.txt --val_file /path/to/your/val.txt --eval --apply_sigmoid

def preprocess_image(image_path, target_size=(224, 224), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """预处理图像用于模型推理"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 调整大小
    img = cv2.resize(img, target_size)
    
    # 归一化
    img = img.astype(np.float32) / 255.0
    img = (img - np.array(mean)) / np.array(std)
    
    # HWC -> CHW
    img = img.transpose(2, 0, 1)
    
    # 添加批次维度并确保类型为float32
    img = np.expand_dims(img, 0).astype(np.float32)
    
    return img

def load_labels(label_path):
    """加载类别标签"""
    with open(label_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def load_val_annotations(val_file):
    """加载验证集的多标签标注信息，格式：图像路径 标签1,标签2,...,标签N"""
    annotations = {}
    full_path = os.path.abspath(val_file)
    if not os.path.exists(full_path):
        print(f"警告: 验证集标注文件不存在: {full_path}")
        return annotations
        
    print(f"加载验证集标注文件: {full_path}")
    with open(full_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')  # 使用制表符分隔
            if len(parts) >= 2:
                image_path = parts[0]
                # 提取文件名作为键
                image_name = os.path.basename(image_path)
                
                # 解析多标签one-hot编码
                label_values = parts[1].split(',')
                labels = np.array([int(v) for v in label_values])
                
                annotations[image_name] = labels
    
    print(f"成功加载 {len(annotations)} 条验证集标注")
    return annotations

def calculate_multilabel_metrics(predictions, annotations, num_classes, vehicle_type_classes=10, color_classes=11):
    """计算多标签分类的评估指标"""
    # 初始化统计变量
    correct_by_class = np.zeros(num_classes)
    total_gt_by_class = np.zeros(num_classes)
    total_pred_by_class = np.zeros(num_classes)
    
    # 统计各个类别的TP, FP, FN
    for image_name, pred_labels in predictions.items():
        if image_name in annotations:
            true_labels = annotations[image_name]
            
            # 对于每个类别
            for c in range(num_classes):
                if true_labels[c] == 1:  # GT中有此类别
                    total_gt_by_class[c] += 1
                    if pred_labels[c] == 1:  # 预测也有此类别
                        correct_by_class[c] += 1
                if pred_labels[c] == 1:  # 预测有此类别
                    total_pred_by_class[c] += 1
    
    # 计算每个类别的precision和recall
    precision_per_class = np.zeros(num_classes)
    recall_per_class = np.zeros(num_classes)
    
    for i in range(num_classes):
        # 准确率 = TP / (TP + FP)
        if total_pred_by_class[i] > 0:
            precision_per_class[i] = correct_by_class[i] / total_pred_by_class[i]
        
        # 召回率 = TP / (TP + FN)
        if total_gt_by_class[i] > 0:
            recall_per_class[i] = correct_by_class[i] / total_gt_by_class[i]
    
    # 计算总体指标
    total_correct = correct_by_class.sum()
    total_gt = total_gt_by_class.sum()
    total_pred = total_pred_by_class.sum()
    
    micro_precision = total_correct / total_pred if total_pred > 0 else 0
    micro_recall = total_correct / total_gt if total_gt > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    

    print(f"===每类的准确率矩阵===")
    print(precision_per_class)
    print(f"===每类的召回率矩阵===")

    print(recall_per_class)
    # 计算宏平均
    macro_precision = np.mean(precision_per_class)
    macro_recall = np.mean(recall_per_class)
    
    # 计算加权平均
    class_weights = total_gt_by_class / total_gt if total_gt > 0 else np.zeros(num_classes)
    weighted_precision = np.sum(precision_per_class * class_weights)
    weighted_recall = np.sum(recall_per_class * class_weights)
    
    # 修改计算车型和颜色的单独指标，忽略样本数量为0的类别
    vehicle_type_precisions = []
    vehicle_type_recalls = []
    for i in range(vehicle_type_classes):
        if total_gt_by_class[i] > 0:  # 只统计有样本的类别
            vehicle_type_recalls.append(recall_per_class[i])
        if total_pred_by_class[i] > 0:  # 只统计有预测的类别
            vehicle_type_precisions.append(precision_per_class[i])
            
    color_precisions = []
    color_recalls = []
    for i in range(vehicle_type_classes, vehicle_type_classes + color_classes):
        if total_gt_by_class[i] > 0:  # 只统计有样本的类别
            color_recalls.append(recall_per_class[i])
        if total_pred_by_class[i] > 0:  # 只统计有预测的类别
            color_precisions.append(precision_per_class[i])
    
    vehicle_type_metrics = {
        'precision': np.mean(vehicle_type_precisions) if vehicle_type_precisions else 0,
        'recall': np.mean(vehicle_type_recalls) if vehicle_type_recalls else 0,
        'total_gt': np.sum(total_gt_by_class[:vehicle_type_classes]),
        'total_pred': np.sum(total_pred_by_class[:vehicle_type_classes]),
        'total_correct': np.sum(correct_by_class[:vehicle_type_classes])
    }
    
    color_metrics = {
        'precision': np.mean(color_precisions) if color_precisions else 0,
        'recall': np.mean(color_recalls) if color_recalls else 0,
        'total_gt': np.sum(total_gt_by_class[vehicle_type_classes:]),
        'total_pred': np.sum(total_pred_by_class[vehicle_type_classes:]),
        'total_correct': np.sum(correct_by_class[vehicle_type_classes:])
    }
    
    return {
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'class_samples': total_gt_by_class,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'vehicle_type_metrics': vehicle_type_metrics,
        'color_metrics': color_metrics
    }

class ONNXPredictor:
    def __init__(self, model_path, label_path=None, apply_sigmoid=True):
        """初始化ONNX预测器"""
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.labels = load_labels(label_path) if label_path else None
        self.apply_sigmoid = apply_sigmoid  # 是否应用sigmoid
        
        # 获取输入尺寸
        if len(self.input_shape) == 4:
            _, _, self.height, self.width = self.input_shape
        else:
            self.height, self.width = 224, 224  # 默认尺寸
        
    def predict(self, image_path, topk=5, threshold=0.5):
        """对单张图像进行预测，支持多标签分类"""
        # 预处理
        input_data = preprocess_image(
            image_path, 
            target_size=(self.width, self.height)
        )
        
        # 推理
        start_time = time.time()
        outputs = self.session.run(None, {self.input_name: input_data})
        inference_time = time.time() - start_time
        
        # 处理结果 - 假设输出是每个类别的概率
        probs = outputs[0][0]
        
        # 如果需要，应用sigmoid
        if self.apply_sigmoid:
            probs = 1 / (1 + np.exp(-probs))
        
        # 确保每个样本只预测一个车型和一个颜色
        vehicle_type_count = 10  # 假设前10个类别为车型
        color_count = 11         # 假设后11个类别为颜色
        
        # 找到概率最高的车型和颜色
        vehicle_probs = probs[:vehicle_type_count]
        color_probs = probs[vehicle_type_count:vehicle_type_count+color_count]
        
        # 初始化预测标签全为0
        pred_labels = np.zeros_like(probs, dtype=int)
        
        # 如果车型概率最大值超过阈值，则选择该车型
        if np.max(vehicle_probs) > threshold:
            best_vehicle_idx = np.argmax(vehicle_probs)
            pred_labels[best_vehicle_idx] = 1
        
        # 如果颜色概率最大值超过阈值，则选择该颜色
        if np.max(color_probs) > threshold:
            best_color_idx = np.argmax(color_probs)
            pred_labels[vehicle_type_count + best_color_idx] = 1
        
        # 获取概率最高的topk个类别
        top_indices = np.argsort(probs)[::-1][:topk]
        
        results = []
        for idx in top_indices:
            label = self.labels[idx] if self.labels else str(idx)
            results.append((label, float(probs[idx])))
        
        return results, inference_time, pred_labels, probs
    
    def batch_predict(self, image_dir, topk=5, threshold=0.5):
        """对文件夹中的所有图像进行批量预测，支持多标签分类"""
        results = {}
        predictions = {}  # 用于存储每个图像的多标签预测结果
        raw_probs = {}    # 存储原始概率值
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        
        # 确保目录存在
        if not os.path.exists(image_dir):
            print(f"错误: 图像目录不存在: {image_dir}")
            return results, predictions, raw_probs
            
        print(f"搜索图像文件夹: {image_dir}")
        
        # 检查是否有val子目录
        val_dir = os.path.join(image_dir, 'val')
        if os.path.exists(val_dir) and os.path.isdir(val_dir):
            print(f"找到val子目录: {val_dir}")
            image_dir = val_dir
        
        # 获取所有图像文件
        image_files = []
        for root, _, files in os.walk(image_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_exts):
                    rel_path = os.path.relpath(os.path.join(root, file), image_dir)
                    image_files.append(rel_path)
        
        print(f"在目录中找到 {len(image_files)} 个图像文件")
        
        if not image_files:
            print(f"警告: 在 {image_dir} 中未找到任何图像文件")
            return results, predictions, raw_probs
            
        for image_file in tqdm(image_files, desc="处理图像"):
            image_path = os.path.join(image_dir, image_file)
            try:
                preds, infer_time, pred_labels, probs = self.predict(image_path, topk, threshold)
                # 使用基本文件名作为键
                base_name = os.path.basename(image_file)
                results[base_name] = {
                    'predictions': preds,
                    'time': infer_time
                }
                
                # 存储多标签预测结果
                predictions[base_name] = pred_labels
                raw_probs[base_name] = probs
                
            except Exception as e:
                print(f"处理 {image_path} 时出错: {str(e)}")
                
        return results, predictions, raw_probs

def main():
    parser = argparse.ArgumentParser(description='ONNX模型推理工具')
    parser.add_argument('--model', required=True, help='ONNX模型文件路径')
    parser.add_argument('--image', help='输入图像路径')
    parser.add_argument('--dir', help='包含多张图像的文件夹路径')
    parser.add_argument('--labels', help='标签文件路径')
    parser.add_argument('--topk', type=int, default=5, help='返回前k个预测结果')
    parser.add_argument('--val_file', help='验证集标注文件路径(val.txt)，用于评估')
    parser.add_argument('--eval', action='store_true', help='是否进行模型评估')
    parser.add_argument('--threshold', type=float, default=0.5, help='多标签分类的阈值')
    # parser.add_argument('--verbose', action='store_true', help='显示详细调试信息')
    parser.add_argument('--apply_sigmoid', action='store_true', help='是否对模型输出应用sigmoid')
    args = parser.parse_args()
    
    if not args.image and not args.dir:
        parser.error("请提供--image或--dir参数")
    
    # 初始化预测器
    predictor = ONNXPredictor(args.model, args.labels, apply_sigmoid=args.apply_sigmoid)
    
    # 单图推理
    if args.image:
        results, infer_time, pred_labels, _ = predictor.predict(args.image, args.topk, args.threshold)
        print(f"推理时间: {infer_time*1000:.2f}ms")
        print("预测结果:")
        for i, (label, prob) in enumerate(results):
            print(f"{i+1}. {label}: {prob:.6f}")
            
        # 显示预测的标签
        if predictor.labels:
            print("\n预测的标签:")
            vehicle_types = []
            colors = []
            for i, is_present in enumerate(pred_labels):
                if is_present:
                    label = predictor.labels[i]
                    if i < 10:  # 车型
                        vehicle_types.append(label)
                    else:  # 颜色
                        colors.append(label)
            
            print(f"车型: {', '.join(vehicle_types) if vehicle_types else '无匹配'}")
            print(f"颜色: {', '.join(colors) if colors else '无匹配'}")
    
    # 批量推理
    if args.dir:
        print(f"开始批量推理: {args.dir}")
        results, predictions, raw_probs = predictor.batch_predict(args.dir, args.topk, args.threshold)
        print(f"共处理 {len(results)} 张图像")
        
        # 计算平均推理时间
        avg_time = sum(r['time'] for r in results.values()) / len(results) if results else 0
        print(f"平均推理时间: {avg_time*1000:.2f}ms")
        
        # 打印前几个结果作为示例
        for i, (img_name, result) in enumerate(list(results.items())[:3]):
            print(f"\n图像: {img_name}")
            print(f"推理时间: {result['time']*1000:.2f}ms")
            print("预测结果:")
            for j, (label, prob) in enumerate(result['predictions']):
                print(f"{j+1}. {label}: {prob:.6f}")
            
            # 显示预测的标签
            if predictor.labels:
                print("\n预测的标签:")
                vehicle_types = []
                colors = []
                for j, is_present in enumerate(predictions[img_name]):
                    if is_present:
                        label = predictor.labels[j]
                        if j < 10:  # 车型
                            vehicle_types.append(label)
                        else:  # 颜色
                            colors.append(label)
                
                print(f"车型: {', '.join(vehicle_types) if vehicle_types else '无匹配'}")
                print(f"颜色: {', '.join(colors) if colors else '无匹配'}")
                
            if i >= 2 and len(results) > 3:
                print("\n...")
                break
                
        # 评估模型性能
        if args.eval and args.val_file and predictor.labels:
            print(f"开始评估模型性能...")
            # 加载验证集标注
            annotations = load_val_annotations(args.val_file)
            if not annotations:
                print("无法加载验证集标注信息或标注文件为空")
                return
                
            # 显示预测集和标注集的大小
            common_keys = set(predictions.keys()) & set(annotations.keys())
            print(f"预测集大小: {len(predictions)}, 标注集大小: {len(annotations)}")
            print(f"两者共有的图像数量: {len(common_keys)}")
            
            if len(common_keys) == 0:
                print("错误: 预测集和标注集没有共同的图像，无法评估")
                
                # 打印前几个键用于调试
                print("预测集中的前10个文件名:")
                for k in list(predictions.keys())[:10]:
                    print(f"  - {k}")
                    
                print("标注集中的前10个文件名:")
                for k in list(annotations.keys())[:10]:
                    print(f"  - {k}")
                return
                
            # 计算多标签评估指标
            num_classes = len(predictor.labels)
            metrics = calculate_multilabel_metrics(predictions, annotations, num_classes)
                
            # 打印评估结果
            print("\n===== 多标签分类评估结果 =====")
            print(f"微平均准确率: {metrics['micro_precision']:.4f}")
            print(f"微平均召回率: {metrics['micro_recall']:.4f}")
            print(f"微平均F1分数: {metrics['micro_f1']:.4f}")
            print(f"宏平均准确率: {metrics['macro_precision']:.4f}")
            print(f"宏平均召回率: {metrics['macro_recall']:.4f}")
            print(f"加权平均准确率: {metrics['weighted_precision']:.4f}")
            print(f"加权平均召回率: {metrics['weighted_recall']:.4f}")
            
            # 打印车型和颜色的单独结果
            vm = metrics['vehicle_type_metrics']
            cm = metrics['color_metrics']
            
            print("\n车型分类结果:")
            print(f"  准确率: {vm['precision']:.4f}")
            print(f"  召回率: {vm['recall']:.4f}")
            print(f"  正确预测数: {vm['total_correct']}")
            print(f"  总真实标签数: {vm['total_gt']}")
            print(f"  总预测标签数: {vm['total_pred']}")
            
            print("\n颜色分类结果:")
            print(f"  准确率: {cm['precision']:.4f}")
            print(f"  召回率: {cm['recall']:.4f}")
            print(f"  正确预测数: {cm['total_correct']}")
            print(f"  总真实标签数: {cm['total_gt']}")
            print(f"  总预测标签数: {cm['total_pred']}")
            
            # 打印每个类别的性能指标:
            print("\n每个类别的性能指标:")
            for i in range(num_classes):
                if metrics['class_samples'][i] > 0:  # 只显示有样本的类别
                    class_name = predictor.labels[i]
                    print(f"类别 {i} ({class_name}):")
                    print(f"  准确率: {metrics['precision_per_class'][i]:.4f}")
                    print(f"  召回率: {metrics['recall_per_class'][i]:.4f}")
                    print(f"  样本数: {int(metrics['class_samples'][i])}")

if __name__ == '__main__':
    print(f"ONNX Runtime 版本: {ort.__version__}")
    print(f"当前工作目录: {os.getcwd()}")
    main()
