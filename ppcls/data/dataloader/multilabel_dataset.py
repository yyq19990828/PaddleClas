#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import numpy as np
import os
import cv2

from ppcls.data.preprocess import transform
from ppcls.utils import logger

from .common_dataset import CommonDataset


class MultiLabelDataset(CommonDataset):
    def _load_anno(self, label_ratio=False):
        self.label_ratio = label_ratio
        self.images = []
        self.labels = []
        
        # 处理cls_path和img_root可能是列表的情况
        cls_paths = self._cls_path if isinstance(self._cls_path, list) else [self._cls_path]
        img_roots = self._img_root if isinstance(self._img_root, list) else [self._img_root]
        
        # 如果只有一个是列表，则扩展另一个为相同长度的列表
        if len(cls_paths) == 1 and len(img_roots) > 1:
            cls_paths = cls_paths * len(img_roots)
        elif len(img_roots) == 1 and len(cls_paths) > 1:
            logger.error("cls_path的大于1的情况下，img_root必须是列表")
            assert len(cls_paths) == len(img_roots), "cls_path的大于1的情况下，img_root必须是列表"
        
        # 确保两个列表长度相等
        assert len(cls_paths) == len(img_roots), "cls_path和img_root列表长度必须相等"
        
        logger.info("数据集由{}个文件夹组成".format(len(cls_paths)))
        # 遍历每对路径组合
        for cls_path, img_root in zip(cls_paths, img_roots):
            # 增强对cls_path路径的判断
            abs_cls_path = cls_path  # 假设为绝对路径
            
            # 获取img_root的目录
            # 构建相对路径
            potential_rel_path = os.path.join(img_root, os.path.basename(cls_path))
            
            abs_exists = os.path.exists(abs_cls_path)
            rel_exists = os.path.exists(potential_rel_path)
            
            # 如果两种方式都存在，优先使用相对路径
            if rel_exists:
                current_cls_path = potential_rel_path
                logger.info(f"使用相对路径 '{potential_rel_path}'")
            elif abs_exists:
                # 保持原来的绝对路径
                current_cls_path = abs_cls_path
                logger.info(f"使用绝对路径 '{abs_cls_path}'")
            else:
                # 两种路径都不存在
                logger.error(f"无法找到标签文件: 绝对路径 '{abs_cls_path}' 或相对路径 '{potential_rel_path}' 都不存在")
                continue  # 跳过这对无效的路径组合
            
            # 确保当前选择的路径存在
            if not os.path.exists(current_cls_path) or not os.path.exists(img_root):
                logger.warning(f"路径不存在: cls_path='{current_cls_path}' 或 img_root='{img_root}'")
                continue
                
            # 加载当前路径组合的数据
            with open(current_cls_path) as fd:
                lines = fd.readlines()
                for l in lines:
                    l = l.strip().split("\t")
                    self.images.append(os.path.join(img_root, l[0]))

                    labels = l[1].split(',')
                    labels = [np.int64(i) for i in labels]

                    self.labels.append(labels)
                    if not os.path.exists(self.images[-1]):
                        logger.warning(f"图像文件不存在: {self.images[-1]}")
        
        # 确保至少加载了一些数据
        assert len(self.images) > 0, "未能加载任何有效的图像和标签数据"
        logger.info("成功加载了{}个图像和标签数据".format(len(self.images)))
        
        if self.label_ratio is not False:
            return np.array(self.labels).mean(0).astype("float32")

    def __getitem__(self, idx):
        try:
            with open(self.images[idx], 'rb') as f:
                img = f.read()
            if self._transform_ops:
                img = transform(img, self._transform_ops)
            img = img.transpose((2, 0, 1))
            label = np.array(self.labels[idx]).astype("float32")
            if self.label_ratio is not False:
                return (img, np.array([label, self.label_ratio]))
            else:
                return (img, label)

        except Exception as ex:
            logger.error("Exception occured when parse line: {} with msg: {}".
                         format(self.images[idx], ex))
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)
