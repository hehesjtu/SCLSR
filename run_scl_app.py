import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import re
import cv2
import torch
import numpy as np
import collections
import random
from types import SimpleNamespace


from SCL import WaveFormerSR

# ---------- 固定随机性 ----------
def _fix_determinism(seed: int = 1234):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    try: torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception: pass
_fix_determinism(1234)

# ---------- 权重路径与倍率 ----------
def _find_weight():
    candidates = [
        os.path.join(os.getcwd(), 'SCL_x4.pth'), 'SCL_x4.pth',
    ]
    for p in candidates:
        if os.path.isfile(p): return p
    raise FileNotFoundError('未找到权重文件（SCL_x4.pth）')

def _infer_scale_from_name(path: str) -> int:
    m = re.search(r'_x([234])\\.pt$', os.path.basename(path).lower())
    return int(m.group(1)) if m else 4

# ---------- 权重映射 ----------
def _remap_weight_norm_ckpt_to_model(ckpt, model_state):
    target = set(model_state.keys()); new = collections.OrderedDict(); used=set()
    bases = {}
    for k in ckpt:
        if k.endswith('.weight_g'): bases.setdefault(k[:-9], {})['g'] = k
        elif k.endswith('.weight_v'): bases.setdefault(k[:-9], {})['v'] = k
    for base, d in bases.items():
        gk, vk = d.get('g'), d.get('v')
        if base + '.weight' in target and gk and vk:
            vparam = ckpt[vk].clone(); g = ckpt[gk].clone()
            out_dim = vparam.shape[0]
            g = g.view(out_dim, *([1] * (vparam.ndim - 1)))
            v_flat = vparam.view(out_dim, -1); v_norm = torch.norm(v_flat, dim=1, keepdim=True).view(out_dim, *([1]*(vparam.ndim-1)))
            new[base + '.weight'] = vparam * (g / (v_norm + 1e-12))
            used.update([gk, vk])
    for k, v in ckpt.items():
        if k not in used and k in target: new[k] = v
    for k, v in model_state.items():
        if k not in new: new[k] = v
    return new

# ---------- 模型加载 ----------
def _load_model(device='cuda'):
    device = 'cuda' if (device=='cuda' and torch.cuda.is_available()) else 'cpu'
    wpath = _find_weight(); scale = _infer_scale_from_name(wpath)
    args = SimpleNamespace(n_colors=3)
    model = WaveFormerSR(img_size=64, scale=scale)
    raw = torch.load(wpath, map_location='cpu')
    ckpt = raw['state_dict'] if isinstance(raw, dict) and 'state_dict' in raw else raw
    remapped = _remap_weight_norm_ckpt_to_model(ckpt, model.state_dict())
    model.load_state_dict(remapped, strict=True)
    model.to(device).eval()
    print(f"[model] device={device}, scale=x{scale}, weight={os.path.basename(wpath)}")
    return model, device, scale

_MODEL, _DEVICE, _SCALE = _load_model('cuda')

# ---------- 工具（0~255 域） ----------
def _to_tensor_nchw255(rgb_u8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(rgb_u8.astype(np.float32)).permute(2,0,1).unsqueeze(0)

def _tensor_to_u8_auto(x: torch.Tensor) -> np.ndarray:
    y = x.detach().cpu().squeeze(0); y_max = float(y.max())
    if y_max <= 1.2:
        img = (y.clamp(0,1).permute(1,2,0).numpy()*255.0 + 0.5).astype(np.uint8)
    else:
        img = y.clamp(0,255).permute(1,2,0).numpy(); img = np.clip(img+0.5,0,255).astype(np.uint8)
    return img

def _nearest_up(rgb_u8: np.ndarray, S:int) -> np.ndarray:
    h,w = rgb_u8.shape[:2]
    return cv2.resize(rgb_u8, (w*S, h*S), interpolation=cv2.INTER_NEAREST)

def _bicubic_down(rgb_u8: np.ndarray, S:int):
    h,w = rgb_u8.shape[:2]; dw, dh = max(1,w//S), max(1,h//S)
    lr = cv2.resize(rgb_u8, (dw,dh), interpolation=cv2.INTER_CUBIC)
    return lr, (w,h)

# ---------- 推理 ----------
@torch.no_grad()
def scl_enhance(img_rgb: np.ndarray, progress=None, return_debug=False):
    S = int(_SCALE)
    if progress: progress.setValue(5)

    nn_up = _nearest_up(img_rgb, S)
    if progress: progress.setValue(20)

    x = _to_tensor_nchw255(img_rgb).to(_DEVICE)
    y = _MODEL(x)
    sr_from_input = _tensor_to_u8_auto(y)

    if sr_from_input.shape[:2] != nn_up.shape[:2]:
        sr_from_input = cv2.resize(sr_from_input, (nn_up.shape[1], nn_up.shape[0]), interpolation=cv2.INTER_AREA)
    if progress: progress.setValue(55)

    # (3) 高频层：增强 3 倍的 |nearest↑S - model↑S|
    highfreq = cv2.absdiff(nn_up, sr_from_input)
    highfreq = np.clip(highfreq * 10, 0, 255).astype(np.uint8)
    if progress: progress.setValue(75)

    lr_bic, (W0,H0) = _bicubic_down(img_rgb, S)
    x_eval = _to_tensor_nchw255(lr_bic).to(_DEVICE)
    y_eval = _MODEL(x_eval)
    sr_eval = _tensor_to_u8_auto(y_eval)
    if (sr_eval.shape[1]!=W0) or (sr_eval.shape[0]!=H0):
        sr_eval = cv2.resize(sr_eval, (W0,H0), interpolation=cv2.INTER_AREA)
    if progress: progress.setValue(95)

    if return_debug:
        dbg = {
            'for_original_window': nn_up,
            'sr_from_input_x4': sr_from_input,
            'highfreq_abs': highfreq,
            'metric_ref_input': img_rgb,
            'metric_pred_enhanced': sr_eval,
        }
        return sr_eval, dbg
    else:
        return sr_eval

scl_enhance.current_scale = int(_SCALE)

if __name__ == '__main__':
    from ui2 import run_app
    run_app(scl_enhance)
