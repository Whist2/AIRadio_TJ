import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from scipy.signal import windows, savgol_filter
from scipy.ndimage import gaussian_filter1d
import os

# === 参数设置 ===
cfile_dir = "cfile_test"
output_dir = "spectrum_images_nolabel"
os.makedirs(output_dir, exist_ok=True)

samp_rate = 100e6
center_freq = 2450e6
n_fft = 8192
n_avg = 8
window_type = "blackmanharris"  # 可选: hann, hamming, flattop, kaiser
smooth_type = "gaussian"        # 可选: none, gaussian, moving_average, savgol

# === 平均频谱计算函数 ===
def averaged_spectrum(iq_data, n_fft, n_avg, window_type, smooth, kaiser_beta=14):
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
        spectrum_db = savgol_filter(spectrum_db, 11, 2)

    return spectrum_db

# === 遍历所有 .cfile 文件 ===
for filename in sorted(os.listdir(cfile_dir)):
    if not filename.endswith(".cfile"):
        continue

    file_path = os.path.join(cfile_dir, filename)
    iq_data = np.fromfile(file_path, dtype=np.complex64)

    if len(iq_data) < n_fft * n_avg:
        print(f"[跳过] 数据不足: {filename}")
        continue

    spectrum_db = averaged_spectrum(iq_data, n_fft, n_avg, window_type, smooth_type)
    freq_axis = np.linspace(center_freq - samp_rate / 2,
                            center_freq + samp_rate / 2,
                            n_fft) / 1e6  # MHz

    # 绘图
    plt.figure(figsize=(16, 6))
    plt.plot(freq_axis, spectrum_db, color='black', linewidth=1.2)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(f"Spectrum - {filename}")
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"{filename.replace('.cfile', '.png')}")
    plt.savefig(out_path)
    plt.close()

    print(f"[已保存] {out_path}")

print("✅ 所有频谱图绘制完成（无标签）")
