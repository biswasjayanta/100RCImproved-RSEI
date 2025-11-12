import numpy as np
import rasterio
from scipy.stats import entropy
from skimage.util import view_as_windows
import matplotlib.pyplot as plt

# ---------- 1. Load and Normalize Input Rasters ----------
def load_and_normalize(path, inverse=False):
    with rasterio.open(path) as src:
        data = src.read(1).astype('float32')
        profile = src.profile
        data[data == src.nodata] = np.nan
    min_val, max_val = np.nanmin(data), np.nanmax(data)
    norm = (max_val - data) / (max_val - min_val) if inverse else (data - min_val) / (max_val - min_val)
    return norm, profile

# ---------- 2. Custom Entropy Function ----------
def custom_entropy(p):
    p = p[~np.isnan(p)]
    if len(p) == 0 or np.sum(p) == 0:
        return np.nan
    p = p / np.sum(p)
    return -np.sum(p * np.log(p)) / np.log(len(p))

def compute_entropy_weights(window):
    window = window[0, :, :, :]
    k, k_val, b = window.shape
    flat = window.reshape(k * k_val, b).T
    center = window[k // 2, k_val // 2, :]

    entropies = []
    for i in range(b):
        values = flat[i]
        values = values[~np.isnan(values)]
        if len(values) == 0:
            entropies.append(np.nan)
        else:
            hist, _ = np.histogram(values, bins='auto')
            probs = hist / np.sum(hist) if np.sum(hist) > 0 else np.zeros_like(hist)
            entropies.append(custom_entropy(probs))

    E = np.array(entropies)
    if np.sum(np.isnan(center)) > 1:
        return np.nan
    center = np.nan_to_num(center)

    if np.any(np.isnan(E)) or np.sum(E) >= b:
        W = np.ones_like(E) / b
    else:
        denom = (b - np.sum(E))
        W = (1 - E) / denom if denom != 0 else np.ones_like(E) / b
    return np.dot(W, center)

# ---------- 3. Sliding Window RSEIFE ----------
def compute_rseife_map(stack, kernel_size=133):
    pad = kernel_size // 2
    padded = np.pad(stack, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=np.nan)
    windows = view_as_windows(padded, (kernel_size, kernel_size, stack.shape[2]))
    rows, cols = windows.shape[:2]
    rseife = np.full((rows, cols), np.nan, dtype='float32')

    for i in range(rows):
        for j in range(cols):
            rseife[i, j] = compute_entropy_weights(windows[i, j])
    return rseife

# ---------- 4. Normalize Function ----------
def normalize(data):
    return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))

# ---------- 5. Yearly RSEIFE Processing ----------
def process_rseife_for_year(year):
    suffix = str(year)[-2:]
    paths = {
        'CVI':     (f'/home/jbiswas/GEE_RSEI/nagpur/CVI_{suffix}.tif', False),
        'Wetness':(f'/home/jbiswas/GEE_RSEI/nagpur/Wetness_{suffix}.tif', False),
        'NDBSI':  (f'/home/jbiswas/GEE_RSEI/nagpur/NDBSI_{suffix}.tif', True),
        'ISI':    (f'/home/jbiswas/GEE_RSEI/nagpur/ISI_{suffix}.tif', True),
        'UI':     (f'/home/jbiswas/GEE_RSEI/nagpur/UI_{suffix}.tif', True),
        'LST':    (f'/home/jbiswas/GEE_RSEI/nagpur/LST_{suffix}.tif', True),
    }

    bands = []
    for _, (path, inverse) in paths.items():
        band, prof = load_and_normalize(path, inverse)
        bands.append(band)

    stack = np.stack(bands, axis=-1)
    rseife = compute_rseife_map(stack, kernel_size=133)
    rseife_norm = normalize(rseife)

    prof.update(dtype='float32', count=1, height=rseife_norm.shape[0], width=rseife_norm.shape[1], nodata=np.nan)
    with rasterio.open(f"RSEIFE_nag_{suffix}.tif", 'w', **prof) as dst:
        dst.write(rseife_norm[np.newaxis, :, :].astype('float32'))

# ---------- 6. Run for All Years ----------
for year in [24, 22, 20, 18, 16]:
    process_rseife_for_year(year)
