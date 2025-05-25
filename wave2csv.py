# -*- coding: utf-8 -*-
"""
Created on Sun May 25 11:29:15 2025

@author: shine
"""

# -*- coding: utf-8 -*-
import numpy as np
from scipy.fft import fft, fftshift
from scipy.signal import windows, savgol_filter
from scipy.ndimage import gaussian_filter1d
import os
import pickle
import sys

# 设置当前目录为脚本路径
os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

# 参数
cfile_dir = "cfile_train"
label_path = "train_labels.pkl"
output_dir = "spectrum_csv"
samp_rate = 100e6
center_freq = 2450e6
n_fft = 8192
n_avg = 8
start_idx = 0

window_type = "blackmanharris"
smooth_type = "gaussian"

os.makedirs(output_dir, exist_ok=True)

# 平均频谱函数
def extract_spectrum_db(iq_data, n_fft, n_avg, window_type, smooth, kaiser_beta=14):
    avg = np.zeros(n_fft, dtype=np.float32)
    window = getattr(windows, window_type)(n_fft) if window_type != "kaiser" else windows.kaiser(n_fft, kaiser_beta)

    for i in range(n_avg):
        start = i * n_fft
        if start + n_fft > len(iq_data):
            break
        segment = iq_data[start:start + n_fft] * window
        spec = fftshift(fft(segment))
        avg += np.abs(spec)
    avg /= n_avg
    spectrum_db = 20 * np.log10(avg + 1e-12)

    # 平滑
    if smooth == "moving_average":
        spectrum_db = np.convolve(spectrum_db, np.ones(5)/5, mode='same')
    elif smooth == "gaussian":
        spectrum_db = gaussian_filter1d(spectrum_db, sigma=2)
    elif smooth == "savgol":
        spectrum_db = savgol_filter(spectrum_db, window_length=11, polyorder=2)

    return spectrum_db

# 读取标签
with open(label_path, "rb") as f:
    train_labels = pickle.load(f)

# 批量处理并保存为 CSV
for idx in range(start_idx, len(train_labels)):
    cfile_path = os.path.join(cfile_dir, f"train_{idx:05d}.cfile")
    if not os.path.exists(cfile_path):
        print(f"[跳过] 文件不存在: {cfile_path}")
        continue

    iq_data = np.fromfile(cfile_path, dtype=np.complex64)
    if len(iq_data) < n_fft * n_avg:
        print(f"[跳过] 数据不足: {cfile_path}")
        continue

    spectrum_db = extract_spectrum_db(iq_data, n_fft, n_avg, window_type, smooth_type)

    # 计算频率轴
    freq_axis = np.linspace(center_freq - samp_rate / 2,
                            center_freq + samp_rate / 2,
                            n_fft) / 1e6  # MHz

    # 拼接频率与功率数据并保存为 CSV
    output_array = np.column_stack((freq_axis, spectrum_db))
    out_path = os.path.join(output_dir, f"spectrum_{idx:05d}.csv")
    np.savetxt(out_path, output_array, fmt="%.6f", delimiter=",", header="Frequency(MHz),Magnitude(dB)", comments='')

    if idx % 100 == 0:
        print(f"[已保存] {idx}/24000")

print("✅ 所有频谱已保存为 CSV 格式（频率 + 幅度）")
