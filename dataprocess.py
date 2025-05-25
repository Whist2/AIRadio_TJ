# -*- coding: utf-8 -*-
"""
Created on Sat May 24 23:07:57 2025

@author: shine
"""

import h5py
import numpy as np
import json
import os
import pickle
import sys
os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

def load_train_data(h5_path):
    """加载训练数据集"""
    with h5py.File(h5_path, 'r') as f:
        waveforms = f['waveforms'][:]  # (24000, 100000)
        labels = f['labels'][:]
        parsed_labels = [np.array(json.loads(label.decode('utf-8'))) for label in labels]
        print(f"训练数据加载完成，共 {len(waveforms)} 个样本")
        return waveforms, parsed_labels

def load_test_data(h5_path):
    """加载测试数据集"""
    with h5py.File(h5_path, 'r') as f:
        waveforms = f['waveforms'][:]  # (1000, 100000)
        questions = f['questions'][:]
        parsed_questions = [q.decode('utf-8') for q in questions]
        print(f"测试数据加载完成，共 {len(waveforms)} 个样本")
        return waveforms, parsed_questions

def export_each_row_to_cfile(waveforms, output_dir, prefix="sample"):
    """将每一行IQ数据保存为一个.cfile文件"""
    os.makedirs(output_dir, exist_ok=True)
    total = waveforms.shape[0]

    for i in range(total):
        iq = waveforms[i].astype(np.complex64)
        file_path = os.path.join(output_dir, f"{prefix}_{i:05d}.cfile")
        iq.tofile(file_path)

        if i % 100 == 0:
            print(f"已保存: {file_path}")

    print(f"共保存 {total} 个 .cfile 文件到: {output_dir}")

if __name__ == "__main__":
    # 设置路径
    train_path = 'train.h5'
    test_path = 'test_public.h5'
    
    train_output_dir = 'cfile_train'
    test_output_dir = 'cfile_test'
    
    # 加载训练数据并导出
    train_waveforms, train_labels = load_train_data(train_path)
    export_each_row_to_cfile(train_waveforms, train_output_dir, prefix="train")

    # 加载测试数据并导出
    test_waveforms, test_questions = load_test_data(test_path)
    export_each_row_to_cfile(test_waveforms, test_output_dir, prefix="test")
    train_labels1=map(list,train_labels)
    train_labels1=list(train_labels1)
    with open("train_labels.pkl", "wb") as f:
        pickle.dump(train_labels, f)