# -*- coding: utf-8 -*-
"""
Created on Sun May 25 00:35:38 2025
@author: shine
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from scipy.signal import windows, savgol_filter
from scipy.ndimage import gaussian_filter1d
import os
import pickle
import sys

# 设置当前目录为脚本路径
os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

# === 参数 ===
cfile_dir = "cfile_train"
label_path = "train_labels.pkl"
output_dir = "spectrum_images"
samp_rate = 100e6
center_freq = 2450e6
n_fft = 8192
n_avg = 8
start_idx = 0

# 滤波选项（可修改）
window_type = "blackmanharris"     # 例如: hann, blackmanharris, flattop
smooth_type = "gaussian"           # 例如: none, moving_average, gaussian, savgol

os.makedirs(output_dir, exist_ok=True)

# === 平均频谱计算 + 绘图函数 ===
def plot_spectrum_with_filters(iq_data, bands=None, title="Spectrum",
                                samp_rate=100e6, center_freq=2450e6,
                                n_fft=8192, n_avg=8,
                                window_type="blackmanharris",
                                smooth="none", kaiser_beta=14,
                                save_path=None):
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

    # 平滑处理
    if smooth == "moving_average":
        spectrum_db = np.convolve(spectrum_db, np.ones(5)/5, mode='same')
    elif smooth == "gaussian":
        spectrum_db = gaussian_filter1d(spectrum_db, sigma=2)
    elif smooth == "savgol":
        spectrum_db = savgol_filter(spectrum_db, window_length=11, polyorder=2)

    # 频率轴
    freq_axis = np.linspace(center_freq - samp_rate/2,
                            center_freq + samp_rate/2,
                            n_fft) / 1e6

    # 绘图
    plt.figure(figsize=(20, 6))
    plt.plot(freq_axis, spectrum_db, color='black', linewidth=1.2, label='Power Spectrum')
    plt.grid(True, linestyle='--', alpha=0.3)

    # 频段标注
    if bands.any():
        ymax = np.max(spectrum_db)
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        for i, (start, end) in enumerate(bands):
            color = colors[i % len(colors)]
            plt.axvline(x=start, color=color, linestyle='--', linewidth=1.2)
            plt.text(start, ymax + 2, f"{start:.1f}", color=color, rotation=90,
                     verticalalignment='bottom', horizontalalignment='center', fontsize=9)
            plt.axvline(x=end, color=color, linestyle='-', linewidth=1.2)
            plt.text(end, ymax + 2, f"{end:.1f}", color=color, rotation=90,
                     verticalalignment='bottom', horizontalalignment='center', fontsize=9)

    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# === 加载标签 ===
with open(label_path, "rb") as f:
    train_labels = pickle.load(f)

# === 批量绘图 ===
for idx, band_list in enumerate(train_labels[start_idx:], start=start_idx):
    cfile_path = os.path.join(cfile_dir, f"train_{idx:05d}.cfile")
    if not os.path.exists(cfile_path):
        print(f"[跳过] 文件不存在: {cfile_path}")
        continue

    iq_data = np.fromfile(cfile_path, dtype=np.complex64)
    if len(iq_data) < n_fft * n_avg:
        print(f"[跳过] 数据不足: {cfile_path}")
        continue

    out_path = os.path.join(output_dir, f"spectrum_{idx:05d}.png")
    plot_spectrum_with_filters(
        iq_data,
        bands=band_list,
        title=f"Spectrum {idx:05d}",
        samp_rate=samp_rate,
        center_freq=center_freq,
        n_fft=n_fft,
        n_avg=n_avg,
        window_type=window_type,
        smooth=smooth_type,
        save_path=out_path
    )

    if idx % 100 == 0:
        print(f"[已完成] {idx}/24000")

print("✅ 批量绘图完成！")
