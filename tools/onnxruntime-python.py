import argparse
import numpy as np
import onnxruntime as ort
import cv2
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil
from sklearn.metrics import confusion_matrix

# 常用的 CLI 启动案例:
#
# 1. 单张图像推理:
# python onnxruntime-python.py --model /path/to/your/model.onnx --image /path/to/your/image.jpg --labels /path/to/your/labels.txt --topk 3
#
# 2. 批量图像推理:
# python onnxruntime-python.py --model /path/to/your/model.onnx --data_dir /path/to/your/image/directory1 /path/to/your/image/directory2 --labels /path/to/your/labels.txt --threshold 0.6
#
# 3. 批量推理并进行评估:
# python onnxruntime-python.py --model /path/to/your/model.onnx --data_dir /path/to/your/image/directory1 /path/to/your/image/directory2 --labels /path/to/your/labels.txt --val_file val.txt --eval --apply_sigmoid

def preprocess_image(image_path, target_size=(224, 224), mean=None, std=None):
    """预处理图像用于模型推理，均值和方差实时计算"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    # 实时计算均值和方差
    # if mean is None:
    #     mean = np.mean(img, axis=(0, 1), keepdims=False)
    # if std is None:
    #     std = np.std(img, axis=(0, 1), keepdims=False)
    #     std[std == 0] = 1e-6  # 防止除以0
    # img = (img - mean) / std
    img = img.transpose(2, 0, 1)
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

def load_multiple_val_annotations(data_dirs, val_file):
    """加载多个验证集的标注信息
    
    Args:
        data_dirs (list): 数据集目录路径列表
        val_file (str): 验证集文件名
        
    Returns:
        dict: 图像路径到标签的映射字典
    """
    combined_annotations = {}

    for data_dir in data_dirs:
        val_path = os.path.join(data_dir, val_file)
        if not os.path.exists(val_path):
            print(f"警告: 验证集标注文件不存在: {val_path}")
            continue

        print(f"加载验证集标注文件: {val_path}")
        with open(val_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ')  # 使用空格分隔
                if len(parts) >= 2:
                    rel_image_path = parts[0]  # 获取图像的相对路径
                    # 构建完整的图像路径，从数据目录开始
                    full_image_path = os.path.join(data_dir, rel_image_path)
                    label_values = parts[1].split(',')
                    labels = np.array([int(v) for v in label_values])
                    # 用完整的图像路径作为键，将标签保存到字典中
                    combined_annotations[full_image_path] = labels

    print(f"成功加载 {len(combined_annotations)} 条验证集标注")
    return combined_annotations

def calculate_multilabel_metrics(predictions, annotations, raw_probs, num_classes, vehicle_type_classes=13, color_classes=11):
    """计算多标签分类的评估指标"""
    # 初始化统计变量
    correct_by_class = np.zeros(num_classes)
    total_gt_by_class = np.zeros(num_classes)
    total_pred_by_class = np.zeros(num_classes)
    
    # 存储用于混淆矩阵计算的数据
    raw_probs_dict = {}
    
    # 统计各个类别的TP, FP, FN
    for image_path, pred_labels in predictions.items():
        # 当使用完整路径作为键时，直接使用键进行查找
        if image_path in annotations:
            true_labels = annotations[image_path]
            
            # 存储原始概率，用于错误分析
            if image_path in raw_probs:
                raw_probs_dict[image_path] = raw_probs[image_path]
            
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
    
    # 计算车型的加权平均指标
    vehicle_total_gt = np.sum(total_gt_by_class[:vehicle_type_classes])
    vehicle_total_pred = np.sum(total_pred_by_class[:vehicle_type_classes]) 
    vehicle_total_correct = np.sum(correct_by_class[:vehicle_type_classes])
    
    vehicle_weighted_precision = vehicle_total_correct / vehicle_total_pred if vehicle_total_pred > 0 else 0
    vehicle_weighted_recall = vehicle_total_correct / vehicle_total_gt if vehicle_total_gt > 0 else 0
    
    vehicle_type_metrics = {
        'precision': np.mean(vehicle_type_precisions) if vehicle_type_precisions else 0,
        'recall': np.mean(vehicle_type_recalls) if vehicle_type_recalls else 0,
        'weighted_precision': vehicle_weighted_precision,
        'weighted_recall': vehicle_weighted_recall,
        'total_gt': vehicle_total_gt,
        'total_pred': vehicle_total_pred,
        'total_correct': vehicle_total_correct
    }
    
    # 计算颜色的加权平均指标
    color_total_gt = np.sum(total_gt_by_class[vehicle_type_classes:])
    color_total_pred = np.sum(total_pred_by_class[vehicle_type_classes:])
    color_total_correct = np.sum(correct_by_class[vehicle_type_classes:])
    
    color_weighted_precision = color_total_correct / color_total_pred if color_total_pred > 0 else 0
    color_weighted_recall = color_total_correct / color_total_gt if color_total_gt > 0 else 0
    
    color_metrics = {
        'precision': np.mean(color_precisions) if color_precisions else 0,
        'recall': np.mean(color_recalls) if color_recalls else 0,
        'weighted_precision': color_weighted_precision,
        'weighted_recall': color_weighted_recall,
        'total_gt': color_total_gt,
        'total_pred': color_total_pred,
        'total_correct': color_total_correct
    }
    
    # 生成混淆矩阵数据
    confusion_data = generate_confusion_matrix(
        predictions, 
        annotations, 
        num_classes, 
        vehicle_type_classes, 
        color_classes
    )
    
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
        'color_metrics': color_metrics,
        'confusion_data': confusion_data,
        'raw_probs_dict': raw_probs_dict
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
        # print(f'读取图片: {image_path}')
        # print(probs)
        
        # 如果需要，应用sigmoid
        if self.apply_sigmoid:
            probs = 1 / (1 + np.exp(-probs))
        
        # 确保每个样本只预测一个车型和一个颜色
        vehicle_type_count = 13  # 13个车型类别 (0-12)
        color_count = 11         # 11个颜色类别 (13-23)
        
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
    
    def batch_predict(self, image_dir, topk=5, threshold=0.5, annotations=None):
        """对文件夹中的所有图像进行批量预测，支持多标签分类
        
        Args:
            image_dir (str): 图像所在的目录（如果使用annotations，此参数可以为任意值）
            topk (int): 返回的top-k个预测结果
            threshold (float): 多标签分类的阈值
            annotations (dict, optional): 包含图像路径和标签的字典，如果提供则只处理这些图像
        
        Returns:
            tuple: 包含预测结果、标签预测和原始概率的字典
        """
        results = {}
        predictions = {}  # 用于存储每个图像的多标签预测结果
        raw_probs = {}    # 存储原始概率值
        
        # 如果提供了annotations，则直接使用这些图像
        if annotations:
            print(f"使用验证集标注中的 {len(annotations)} 张图像进行预测")
            # 直接使用标注中的完整图像路径
            image_files = []
            for img_path in annotations.keys():
                if os.path.exists(img_path):
                    image_files.append(img_path)
                
            print(f"在验证集中找到 {len(image_files)} 个有效图像文件")
        else:
            # 确保目录存在
            if not os.path.exists(image_dir):
                print(f"错误: 图像目录不存在: {image_dir}")
                return results, predictions, raw_probs
            
            print(f"搜索图像文件夹: {image_dir}")
            image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
            # 获取所有图像文件
            image_files = []
            for root, _, files in os.walk(image_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_exts):
                        image_files.append(os.path.join(root, file))

            print(f"在目录中找到 {len(image_files)} 个图像文件")

        if not image_files:
            print(f"警告: 在 {image_dir} 中未找到任何图像文件")
            return results, predictions, raw_probs

        for image_file in tqdm(image_files, desc="处理图像"):
            try:
                preds, infer_time, pred_labels, probs = self.predict(image_file, topk, threshold)
                
                # 如果使用的是annotations，使用完整路径作为键以便与annotations匹配
                if annotations:
                    key = image_file  # 使用完整路径作为键
                else:
                    # 否则使用文件名作为键
                    key = os.path.basename(image_file)
                    
                results[key] = {
                    'predictions': preds,
                    'time': infer_time
                }

                # 存储多标签预测结果
                predictions[key] = pred_labels
                raw_probs[key] = probs

            except Exception as e:
                print(f"处理 {image_file} 时出错: {str(e)}")

        return results, predictions, raw_probs

def generate_confusion_matrix(predictions, annotations, num_classes, vehicle_type_classes=13, color_classes=11, labels=None):
    """生成混淆矩阵"""
    # 分别为车型和颜色创建混淆矩阵
    vehicle_true = []
    vehicle_pred = []
    color_true = []
    color_pred = []
    
    # 用于收集分类错误的图像
    misclassified_images = {
        'vehicle': [],  # 将存储元组 (image_path, true_class, pred_class, probs)
        'color': []     # 将存储元组 (image_path, true_class, pred_class, probs)
    }
    
    for image_path, pred_labels in predictions.items():
        if image_path not in annotations:
            continue
            
        true_labels = annotations[image_path]
        
        # 获取车型和颜色的真实和预测标签
        true_vehicle_idx = np.argmax(true_labels[:vehicle_type_classes]) if np.sum(true_labels[:vehicle_type_classes]) > 0 else -1
        pred_vehicle_idx = np.argmax(pred_labels[:vehicle_type_classes]) if np.sum(pred_labels[:vehicle_type_classes]) > 0 else -1
        
        true_color_idx = np.argmax(true_labels[vehicle_type_classes:vehicle_type_classes+color_classes]) if np.sum(true_labels[vehicle_type_classes:vehicle_type_classes+color_classes]) > 0 else -1
        if true_color_idx != -1:
            true_color_idx += vehicle_type_classes
        pred_color_idx = np.argmax(pred_labels[vehicle_type_classes:vehicle_type_classes+color_classes]) if np.sum(pred_labels[vehicle_type_classes:vehicle_type_classes+color_classes]) > 0 else -1
        if pred_color_idx != -1:
            pred_color_idx += vehicle_type_classes
        
        # 添加到混淆矩阵计算列表
        if true_vehicle_idx != -1 and pred_vehicle_idx != -1:
            vehicle_true.append(true_vehicle_idx)
            vehicle_pred.append(pred_vehicle_idx)
            
            # 如果分类错误，添加到misclassified_images
            if true_vehicle_idx != pred_vehicle_idx:
                misclassified_images['vehicle'].append((
                    image_path,
                    true_vehicle_idx,
                    pred_vehicle_idx,
                    None  # 原始概率将在后续步骤中添加
                ))
                
        if true_color_idx != -1 and pred_color_idx != -1:
            # 减去偏移量以获得正确的颜色类别索引(0-10)
            color_true.append(true_color_idx - vehicle_type_classes)
            color_pred.append(pred_color_idx - vehicle_type_classes)
            
            # 如果分类错误，添加到misclassified_images
            if true_color_idx != pred_color_idx:
                misclassified_images['color'].append((
                    image_path,
                    true_color_idx,
                    pred_color_idx,
                    None  # 原始概率将在后续步骤中添加
                ))
    
    # 计算混淆矩阵
    vehicle_cm = confusion_matrix(
        vehicle_true, 
        vehicle_pred, 
        labels=np.arange(vehicle_type_classes)
    ) if len(vehicle_true) > 0 else np.zeros((vehicle_type_classes, vehicle_type_classes))
    
    color_cm = confusion_matrix(
        color_true, 
        color_pred, 
        labels=np.arange(color_classes)
    ) if len(color_true) > 0 else np.zeros((color_classes, color_classes))
    
    return {
        'vehicle': vehicle_cm,
        'color': color_cm,
        'misclassified': misclassified_images
    }

def visualize_confusion_matrix(cm, class_names, title, save_path):
    """可视化混淆矩阵并保存到指定路径，使用ASCII标签避免中文字体问题"""
    # 强制使用Agg后端，防止字体问题
    import matplotlib
    matplotlib.use('Agg')
    
    num_classes = len(class_names)
    # Adjust figure size dynamically based on the number of classes
    fig_width = max(12, num_classes * 0.6) 
    fig_height = max(10, num_classes * 0.5)
    plt.figure(figsize=(fig_width, fig_height))

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix", fontsize=16)  # 使用英文标题
    plt.colorbar()
    
    # 使用数字索引作为刻度标签
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, range(len(class_names)), rotation=45, ha="right") # Rotate for better fit
    plt.yticks(tick_marks, range(len(class_names)))
    
    # 在每个单元格中显示数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout(pad=1.5) # Add padding to ensure labels are not cut off
    plt.ylabel('True Label Index', fontsize=14)  # 使用英文标签
    plt.xlabel('Predicted Label Index', fontsize=14)  # 使用英文标签
    
    # 保存图像
    plt.savefig(save_path)
    plt.close()
    
    # 创建索引与标签对应的图例文件
    legend_path = os.path.join(os.path.dirname(save_path), f"{os.path.splitext(os.path.basename(save_path))[0]}_legend.txt")
    with open(legend_path, 'w', encoding='utf-8') as f:
        f.write(f"{'索引':<6}{'标签':<15}\n")
        f.write("-" * 25 + "\n")
        for idx, label in enumerate(class_names):
            f.write(f"{idx:<6}{label:<15}\n")
    
    print(f"混淆矩阵已保存到: {save_path}")
    print(f"标签索引对照表已保存到: {legend_path}")

def collect_misclassified_images(misclassified_data, raw_probs, labels, output_dir, topk=3, save_ori_image=False):
    """收集分类错误的图片并处理"""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    vehicle_dir = os.path.join(output_dir, "vehicle_errors")
    color_dir = os.path.join(output_dir, "color_errors")
    os.makedirs(vehicle_dir, exist_ok=True)
    os.makedirs(color_dir, exist_ok=True)
    
    # 如果需要保存原图，创建ori_image文件夹
    if save_ori_image:
        ori_vehicle_dir = os.path.join(output_dir, "ori_image", "vehicle_errors")
        ori_color_dir = os.path.join(output_dir, "ori_image", "color_errors")
        os.makedirs(ori_vehicle_dir, exist_ok=True)
        os.makedirs(ori_color_dir, exist_ok=True)
    
    # 处理车型错误
    process_misclassified_category(
        misclassified_data['vehicle'], 
        raw_probs,
        labels, 
        vehicle_dir, 
        "车型", 
        0,  #NOTE vehicle_start_idx
        13,  # vehicle_classes
        topk,
        ori_vehicle_dir if save_ori_image else None
    )
    
    # 处理颜色错误
    process_misclassified_category(
        misclassified_data['color'], 
        raw_probs,
        labels, 
        color_dir, 
        "颜色", 
        13,  #NOTE color_start_idx
        11,  # color_classes
        topk,
        ori_color_dir if save_ori_image else None
    )
    
    print(f"已收集并处理 {len(misclassified_data['vehicle'])} 张车型错误图片和 {len(misclassified_data['color'])} 张颜色错误图片")
    return {
        'vehicle_count': len(misclassified_data['vehicle']),
        'color_count': len(misclassified_data['color'])
    }

def process_misclassified_category(misclassified_items, raw_probs, labels, output_dir, category_name, start_idx, num_classes, topk=3, ori_image_dir=None):
    """处理特定类别的错误分类图像，文本信息显示在图片右侧"""
    for i, (image_path, true_idx, pred_idx, _) in enumerate(misclassified_items):
        try:
            # 读取原始图像
            img_original = cv2.imread(image_path)
            if img_original is None:
                print(f"无法读取图像: {image_path}")
                continue
            
            # 将图片缩放至512x512
            img = cv2.resize(img_original, (512, 512))
            
            # 获取原始文件名（不带路径和扩展名）
            file_basename = os.path.basename(image_path)
            filename_no_ext = os.path.splitext(file_basename)[0]
            
            # 获取真实和预测的标签名称
            true_label = labels[true_idx] if 0 <= true_idx < len(labels) else "Unknown"
            pred_label = labels[pred_idx] if 0 <= pred_idx < len(labels) else "Unknown"
            
            # 获取原始概率值
            probs = raw_probs.get(image_path, None)
            
            # 准备要显示的文本信息
            info_txt = []
            info_txt.append(f"File: {file_basename}")
            info_txt.append(f"True Class: {true_idx}")
            info_txt.append(f"True Label: {true_label}")
            info_txt.append(f"Pred Class: {pred_idx}")
            info_txt.append(f"Pred Label: {pred_label}")
            
            # 添加概率信息
            if probs is not None:
                # 获取该类别的所有概率
                category_probs = probs[start_idx:start_idx+num_classes]
                # 获取topk索引
                topk_indices = np.argsort(category_probs)[::-1][:topk]
                # 获取这些索引对应的概率值
                topk_probs = category_probs[topk_indices]
                # 转换索引到标签索引
                topk_labels = [labels[start_idx + idx] for idx in topk_indices]
                
                info_txt.append(f"\nTop {topk} Probabilities:")
                for idx_prob, (label_prob, prob_val) in enumerate(zip(topk_labels, topk_probs)):
                    info_txt.append(f"{idx_prob+1}. {label_prob}: {prob_val:.2%}")
            
            # 设置文本显示参数
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6  # Reduced font scale for better fit
            thickness = 1
            font_color = (255, 255, 255)  # 白色文字
            
            # Calculate dynamic line height based on font scale
            (char_w_template, char_h_template), baseline_template = cv2.getTextSize("Tg", font, font_scale, thickness)
            line_height = char_h_template + baseline_template + 5 # Add some spacing between lines

            # Calculate required width for the text based on the longest line
            max_text_line_width = 0
            for line in info_txt:
                (lw, lh), _ = cv2.getTextSize(line, font, font_scale, thickness)
                if lw > max_text_line_width:
                    max_text_line_width = lw
            
            text_horizontal_padding = 20 # Padding on left/right of text within its panel
            # Calculate the width needed for the text panel itself
            calculated_text_panel_width = max_text_line_width + 2 * text_horizontal_padding
            
            # Ensure text_panel_width has a reasonable minimum and maximum
            min_panel_w = 300 
            # Cap panel width, e.g., to original image width or a fixed max like 800
            # Allow it to be at least image width if image is wide, or up to 800px
            img_h, img_w, img_c = img.shape # Get image dimensions here
            
            text_panel_width = max(min_panel_w, calculated_text_panel_width)
            # Cap the panel width to avoid excessively wide images if a single line is extremely long
            # It can be at most the image width (if large) or an absolute max (e.g., 800)
            text_panel_width = min(text_panel_width, max(img_w, 800))


            # Calculate total height needed for the text block
            text_vertical_padding = 20 # Padding on top/bottom of text block
            text_block_height = (len(info_txt) * line_height) + 2 * text_vertical_padding
            
            # 创建一个新的拼接图像（原图 + 文本区）
            final_height = max(img_h, text_block_height)
            final_img = np.zeros((final_height, img_w + text_panel_width, img_c), dtype=np.uint8)
            final_img.fill(0)  # 全黑背景
            
            # 在左侧放置原始图像
            final_img[0:img_h, 0:img_w] = img
            
            # 在右侧放置文本信息（黑底白字）
            # Base y position for the first line of text
            base_char_height_for_y_calc = cv2.getTextSize("Tg", font, font_scale, thickness)[0][1]
            y_start_offset = text_vertical_padding + base_char_height_for_y_calc

            for i_line, line_content in enumerate(info_txt):
                y_pos = y_start_offset + (i_line * line_height)
                cv2.putText(
                    final_img, 
                    line_content, 
                    (img_w + text_horizontal_padding, y_pos),
                    font, 
                    font_scale, 
                    font_color, 
                    thickness
                )
            
            # 保存图像
            save_path = os.path.join(output_dir, f"{filename_no_ext}_T{true_idx}_P{pred_idx}.jpg")
            cv2.imwrite(save_path, final_img)
            
            # 如果需要保存原图
            if ori_image_dir is not None:
                ori_save_path = os.path.join(ori_image_dir, f"{filename_no_ext}_T{true_idx}_P{pred_idx}.jpg")
                cv2.imwrite(ori_save_path, img_original)
            
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='ONNX模型推理工具')
    parser.add_argument('--model', required=True, help='ONNX模型文件路径')
    parser.add_argument('--image', help='输入图像路径')
    parser.add_argument('--data_dir', nargs='+', help='包含多个数据集文件夹的路径列表')
    parser.add_argument('--labels', help='标签文件路径')
    parser.add_argument('--topk', type=int, default=5, help='返回前k个预测结果')
    parser.add_argument('--val_file', help='验证集标注文件名(val.txt)，用于评估')
    parser.add_argument('--eval', action='store_true', help='是否进行模型评估')
    parser.add_argument('--threshold', type=float, default=0.5, help='多标签分类的阈值')
    parser.add_argument('--apply_sigmoid', action='store_true', help='是否对模型输出应用sigmoid')
    parser.add_argument('--confusion_matrix', action='store_true', help='是否生成混淆矩阵')
    parser.add_argument('--error_collection', action='store_true', help='是否收集分类错误的图片')
    parser.add_argument('--save_ori_image', action='store_true', help='是否保存错误图片的原图到ori_image文件夹')
    parser.add_argument('--error_dir', default='dataset/VA/eval_error', help='分类错误图片的保存目录')
    args = parser.parse_args()

    if not args.image and not args.data_dir:
        parser.error("请提供--image或--data_dir参数")

    predictor = ONNXPredictor(args.model, args.labels, apply_sigmoid=args.apply_sigmoid)

    if args.image:
        results, infer_time, pred_labels, _ = predictor.predict(args.image, args.topk, args.threshold)
        print(f"推理时间: {infer_time*1000:.2f}ms")
        print("预测结果:")
        for i, (label, prob) in enumerate(results):
            print(f"{i+1}. {label}: {prob:.6f}")

        if predictor.labels:
            print("\n预测的标签:")
            vehicle_types = []
            colors = []
            for i, is_present in enumerate(pred_labels):
                if is_present:
                    label = predictor.labels[i]
                    if i < 10:
                        vehicle_types.append(label)
                    else:
                        colors.append(label)

            print(f"车型: {', '.join(vehicle_types) if vehicle_types else '无匹配'}")
            print(f"颜色: {', '.join(colors) if colors else '无匹配'}")

    if args.data_dir:
        print(f"开始批量推理: {args.data_dir}")
        combined_annotations = {}
        if args.eval and args.val_file:
            combined_annotations = load_multiple_val_annotations(args.data_dir, args.val_file)

        # 获取ONNX模型文件名（不带扩展名）
        onnx_base = os.path.splitext(os.path.basename(args.model))[0]
        # 分类错误和混淆矩阵的主目录（与--error_dir同级，且以onnx模型名命名）
        output_root = os.path.join(os.path.dirname(args.error_dir), onnx_base)
        print(f"结果将保存到: {output_root}")
        # if os.path.exists(output_root):
        #     shutil.rmtree(output_root)
        os.makedirs(output_root, exist_ok=True)
        # 混淆矩阵和分类错误图片的子目录
        confusion_dir = os.path.join(output_root, "confusion_matrix")
        error_dir = os.path.join(output_root, "misclassified")
        os.makedirs(confusion_dir, exist_ok=True)
        os.makedirs(error_dir, exist_ok=True)

        # 如果是评估模式并且有标注，直接使用所有标注的图像路径进行一次性预测
        if args.eval and combined_annotations:
            print(f"直接使用验证集标注中的图像进行批量预测")
            base_dir = args.data_dir[0]
            results, predictions, raw_probs = predictor.batch_predict(
                base_dir,
                args.topk,
                args.threshold,
                annotations=combined_annotations
            )
        else:
            results, predictions, raw_probs = {}, {}, {}
            for data_dir in args.data_dir:
                dir_results, dir_predictions, dir_raw_probs = predictor.batch_predict(
                    data_dir,
                    args.topk,
                    args.threshold,
                    annotations=None
                )
                results.update(dir_results)
                predictions.update(dir_predictions)
                raw_probs.update(dir_raw_probs)

        # ========== 输出信息写入README.md ========== #
        readme_path = os.path.join(output_root, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(f"共处理 {len(results)} 张图像\n")
            avg_time = sum(r['time'] for r in results.values()) / len(results) if results else 0
            f.write(f"平均推理时间: {avg_time*1000:.2f}ms\n")

            if args.eval and predictor.labels:
                f.write(f"开始评估模型性能...\n")
                if not combined_annotations:
                    f.write("无法加载验证集标注信息或标注文件为空\n")
                    return

                f.write(f"预测集大小: {len(predictions)}, 标注集大小: {len(combined_annotations)}\n")
                if len(predictions) == 0:
                    f.write("错误: 没有找到任何可预测的图像，无法评估\n")
                    return

                num_classes = len(predictor.labels)
                metrics = calculate_multilabel_metrics(predictions, combined_annotations, raw_probs, num_classes)

                f.write("\n===== 多标签分类评估结果 =====\n")
                f.write(f"微平均准确率: {metrics['micro_precision']:.4f}\n")
                f.write(f"微平均召回率: {metrics['micro_recall']:.4f}\n")
                f.write(f"微平均F1分数: {metrics['micro_f1']:.4f}\n")
                f.write(f"宏平均准确率: {metrics['macro_precision']:.4f}\n")
                f.write(f"宏平均召回率: {metrics['macro_recall']:.4f}\n")
                f.write(f"加权平均准确率: {metrics['weighted_precision']:.4f}\n")
                f.write(f"加权平均召回率: {metrics['weighted_recall']:.4f}\n")

                # 生成混淆矩阵
                if args.confusion_matrix:
                    f.write("\n生成混淆矩阵...\n")
                    confusion_data = metrics['confusion_data']
                    vehicle_labels = predictor.labels[:13] #NOTE: 车型标签从0到12(一共13个车型)
                    color_labels = predictor.labels[13:24] #NOTE: 颜色标签从13到23(一共11个颜色)
                    visualize_confusion_matrix(
                        confusion_data['vehicle'],
                        vehicle_labels,
                        "Vehicle Type Confusion Matrix",
                        os.path.join(confusion_dir, "vehicle_confusion_matrix.png")
                    )
                    visualize_confusion_matrix(
                        confusion_data['color'],
                        color_labels,
                        "Color Confusion Matrix",
                        os.path.join(confusion_dir, "color_confusion_matrix.png")
                    )
                    f.write(f"混淆矩阵已保存到: {confusion_dir}\n")

                # 收集分类错误的图片
                if args.error_collection:
                    f.write("\n收集分类错误的图片...\n")
                    misclassified_stats = collect_misclassified_images(
                        metrics['confusion_data']['misclassified'],
                        metrics['raw_probs_dict'],
                        predictor.labels,
                        error_dir,
                        args.topk,
                        args.save_ori_image
                    )
                    f.write(f"分类错误的图片已保存到: {error_dir}\n")
                    f.write(f"车型错误图片数量: {misclassified_stats['vehicle_count']}\n")
                    f.write(f"颜色错误图片数量: {misclassified_stats['color_count']}\n")

                # 打印车型和颜色的单独结果
                vm = metrics['vehicle_type_metrics']
                cm = metrics['color_metrics']

                f.write("\n车型分类结果:\n")
                f.write(f"  准确率(宏平均): {vm['precision']:.4f}\n")
                f.write(f"  召回率(宏平均): {vm['recall']:.4f}\n")
                f.write(f"  准确率(加权): {vm['weighted_precision']:.4f}\n")
                f.write(f"  召回率(加权): {vm['weighted_recall']:.4f}\n")
                f.write(f"  正确预测数: {vm['total_correct']}\n")
                f.write(f"  总真实标签数: {vm['total_gt']}\n")
                f.write(f"  总预测标签数: {vm['total_pred']}\n")

                f.write("\n颜色分类结果:\n")
                f.write(f"  准确率(宏平均): {cm['precision']:.4f}\n")
                f.write(f"  召回率(宏平均): {cm['recall']:.4f}\n")
                f.write(f"  准确率(加权): {cm['weighted_precision']:.4f}\n")
                f.write(f"  召回率(加权): {cm['weighted_recall']:.4f}\n")
                f.write(f"  正确预测数: {cm['total_correct']}\n")
                f.write(f"  总真实标签数: {cm['total_gt']}\n")
                f.write(f"  总预测标签数: {cm['total_pred']}\n")

                f.write("\n每个类别的性能指标:\n")
                for i in range(num_classes):
                    if metrics['class_samples'][i] > 0:
                        class_name = predictor.labels[i]
                        f.write(f"类别 {i} ({class_name}):\n")
                        f.write(f"  准确率: {metrics['precision_per_class'][i]:.4f}\n")
                        f.write(f"  召回率: {metrics['recall_per_class'][i]:.4f}\n")
                        f.write(f"  样本数: {int(metrics['class_samples'][i])}\n")

if __name__ == '__main__':
    print(f"ONNX Runtime 版本: {ort.__version__}")
    print(f"当前工作目录: {os.getcwd()}")
    main()
