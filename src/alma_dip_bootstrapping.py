# alma_dip.py — unified DIP with auto small/large selection + bootstrap uncertainty
# (This file includes: SMALL path, MEMORY path, AUTO selector, and BOOTSTRAP utilities)

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Device helper
# ----------------------------
def pick_device(explicit: Optional[str] = None) -> torch.device:
    if explicit is not None:
        return torch.device(explicit)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------
# Config (small + large + auto selector)
# ----------------------------
@dataclass
class DIPConfig:
    # --- Common optimization ---
    seed: int = 123
    num_iters: int = 2500
    lr: float = 1e-3

    # --- Common regularization ---
    tv_weight: float = 5e-5
    positivity: bool = True

    # --- Common generator (U-Net) ---
    input_depth: int = 32
    base_channels: int = 64
    depth: int = 5
    dropout: float = 0.0

    # --- Common measurement / likelihood ---
    cell_size_arcsec: float = 0.10
    nu: float = 3.0
    learn_sigma: bool = False
    init_sigma: float = 1.0
    out_every: int = 200

    # --- Auto selector ---
    force_mode: str = "auto"          # "auto" | "small" | "memory"
    auto_threshold_n: int = 5_000_000  # choose small if N <= threshold, else memory

    # ====== LARGE / MEMORY PATH OPTIONS (small path ignores these) ======
    batch_vis: int = 16384
    stream_chunk_vis: Optional[int] = None

    # Advanced (u,v) stratified sampler (mixture)
    sampler_type: str = "mixture"
    radial_bins: int = 48
    radial_quantiles: bool = True
    angular_bins: int = 0
    low_k_bins: int = 3
    low_k_boost: float = 3.0

    # Mixture component weights (normalized at runtime)
    alpha_uniform: float = 0.10
    alpha_radial: float = 0.55
    alpha_inv_radius: float = 0.35
    alpha_weight: float = 0.0
    alpha_angular: float = 0.0

    # Inverse-radius component
    inv_radius_gamma: float = 1.0
    radius_eps_frac: float = 1e-6

    # Importance weighting
    importance_snis: bool = True

    # Optional adaptive bin reweighting
    adapt_every: int = 0
    adapt_ema: float = 0.9
    adapt_power: float = 1.0

    # Quantile strategy for huge N
    quantile_mode: str = "auto"      # "auto" | "full" | "sample" | "hist" | "logspace"
    quantile_full_threshold: int = 5_000_000
    quantile_sample_max: int = 2_000_000
    hist_bins_for_quantiles: int = 262_144


# ----------------------------
# TV loss (isotropic)
# ----------------------------
def tv_loss(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    return torch.sum(torch.sqrt(dx * dx + eps)) + torch.sum(torch.sqrt(dy * dy + eps))


# ----------------------------
# Student's-t negative log-likelihood
# ----------------------------
class StudentTLoss(nn.Module):
    def __init__(self, nu: float = 3.0, learn_sigma: bool = False,
                 init_sigma: float = 1.0, reduction: str = "mean"):
        super().__init__()
        if nu <= 0:
            raise ValueError("nu must be > 0")
        self.nu = float(nu)
        self.reduction = reduction
        self.learn_sigma = learn_sigma
        if learn_sigma:
            self.log_sigma = nn.Parameter(torch.log(torch.tensor(float(init_sigma))))
        else:
            self.register_buffer("log_sigma", torch.log(torch.tensor(float(init_sigma))))
        self.d = 2
        self.eps = 1e-8

    def forward(self, pred: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        r = pred - target                             # [N,2]
        r2 = torch.sum(r * r, dim=-1)                # [N]
        if weight is None:
            weight = torch.ones_like(r2)
        sigma2 = torch.exp(2.0 * self.log_sigma)
        scaled = 1.0 + (weight * r2) / (self.nu * sigma2 + self.eps)
        nll = 0.5 * (self.nu + self.d) * torch.log(scaled + self.eps)
        if self.learn_sigma:
            nll = nll + 0.5 * self.d * self.log_sigma
        if self.reduction == "sum":
            return nll.sum()
        elif self.reduction == "mean":
            return nll.mean()
        return nll



# ----------------------------
# DIP generator (robust U-Net)
#  - reflection padding
#  - group norm (BN can be temperamental on small batches)
#  - bilinear upsampling (no transpose conv checkerboards)
# ----------------------------
class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=0, bias=False)
        # Use 8 groups or 1 if channels < 8
        groups = max(1, min(8, out_ch))
        self.gn = nn.GroupNorm(groups, out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.drop = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x)
        x = self.conv(x)
        x = self.gn(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = ConvGNAct(in_ch, out_ch, dropout)
        self.conv2 = ConvGNAct(out_ch, out_ch, dropout)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.conv2(x)
        skip = x
        x = self.pool(x)
        return x, skip


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1 = ConvGNAct(in_ch + skip_ch, out_ch, dropout)
        self.conv2 = ConvGNAct(out_ch, out_ch, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # pad if shapes differ by 1 due to odd dimensions
        dh = skip.size(-2) - x.size(-2)
        dw = skip.size(-1) - x.size(-1)
        if dh != 0 or dw != 0:
            x = F.pad(x, (0, dw, 0, dh))
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DIPUNet(nn.Module):
    def __init__(self, input_depth: int = 32, base_ch: int = 64, depth: int = 5,
                 dropout: float = 0.0, out_ch: int = 1):
        super().__init__()
        self.depth = depth
        chs = [base_ch * min(2 ** i, 8) for i in range(depth)]  # cap growth
        self.downs = nn.ModuleList()
        in_c = input_depth
        for i in range(depth):
            out_c = chs[i]
            self.downs.append(Down(in_c, out_c, dropout))
            in_c = out_c

        self.bottleneck = nn.Sequential(
            ConvGNAct(chs[-1], chs[-1], dropout),
            ConvGNAct(chs[-1], chs[-1], dropout),
        )

        self.ups = nn.ModuleList()
        for i in reversed(range(depth)):
            in_c = chs[i]
            skip_c = chs[i]
            out_c = chs[i - 1] if i > 0 else base_ch
            self.ups.append(Up(in_c, skip_c, out_c, dropout))

        self.final = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch, out_ch, kernel_size=3, padding=0)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        skips = []
        x = z
        for d in self.downs:
            x, s = d(x)
            skips.append(s)
        x = self.bottleneck(x)
        for i, u in enumerate(self.ups):
            x = u(x, skips[-(i + 1)])
        x = self.final(x)
        return x


# =========================================================
# SMALL PATH (original) — precompute full trig tables
# =========================================================
@torch.no_grad()
def _grid_coords(H: int, W: int, cell_size_rad: float, device: torch.device, dtype: torch.dtype):
    l = (torch.arange(W, device=device, dtype=dtype) - (W // 2)) * cell_size_rad  # [W]
    m = (torch.arange(H, device=device, dtype=dtype) - (H // 2)) * cell_size_rad  # [H]
    return l, m

def precompute_uv_phases_full(uv: torch.Tensor, H: int, W: int, cell_size_rad: float,
                              device: torch.device, dtype: torch.dtype = torch.float32):
    u = uv[:, 0].to(device=device, dtype=dtype)
    v = uv[:, 1].to(device=device, dtype=dtype)
    l, m = _grid_coords(H, W, cell_size_rad, device, dtype)
    ul = 2.0 * math.pi * (u[:, None] * l[None, :])  # [N,W]
    vm = 2.0 * math.pi * (v[:, None] * m[None, :])  # [N,H]
    Cul = torch.cos(ul);  Sul = torch.sin(ul)
    Cvm = torch.cos(vm);  Svm = torch.sin(vm)
    return Cul, Sul, Cvm, Svm

def predict_vis_from_precomp(img: torch.Tensor,
                             Cul: torch.Tensor, Sul: torch.Tensor,
                             Cvm: torch.Tensor, Svm: torch.Tensor) -> torch.Tensor:
    I = img[0, 0]
    A = I @ Cul.t()                       # [H,N]
    B = I @ Sul.t()                       # [H,N]
    real = torch.sum(Cvm * A.t(), dim=1) - torch.sum(Svm * B.t(), dim=1)
    imag = -(torch.sum(Cvm * B.t(), dim=1) + torch.sum(Svm * A.t(), dim=1))
    return torch.stack([real, imag], dim=1)  # [N,2]

@torch.no_grad()
def make_dirty_and_psf_full(Cul: torch.Tensor, Sul: torch.Tensor,
                            Cvm: torch.Tensor, Svm: torch.Tensor,
                            re: torch.Tensor, im: torch.Tensor, w: torch.Tensor,
                            normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    H = Cvm.shape[1]; W = Cul.shape[1]
    E = (w * re)[:, None] * Cul - (w * im)[:, None] * Sul
    F = (w * re)[:, None] * Sul + (w * im)[:, None] * Cul
    dirty = Cvm.t() @ E - Svm.t() @ F
    psf   = Cvm.t() @ (w[:, None] * Cul) - Svm.t() @ (w[:, None] * Sul)
    if normalize:
        norm = torch.sum(psf).clamp_min(1e-12)
        dirty = dirty / norm; psf = psf / norm
    return dirty, psf

def reconstruct_dip_small(
    uv: np.ndarray | torch.Tensor,
    vis: Optional[np.ndarray | torch.Tensor] = None,
    weight: Optional[np.ndarray | torch.Tensor] = None,
    img_size: Tuple[int, int] = (256, 256),
    cfg: Optional[DIPConfig] = None,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cfg is None: cfg = DIPConfig()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    dev = pick_device(device)

    def to_t(x):
        if x is None: return None
        if torch.is_tensor(x): return x.to(dev, dtype=torch.float32)
        return torch.as_tensor(x, device=dev, dtype=torch.float32)

    uv_t = to_t(uv)
    if uv_t.ndim != 2: raise ValueError("uv must be 2D")
    if vis is None and weight is None and uv_t.shape[1] >= 5:
        re_t = uv_t[:, 2].clone(); im_t = uv_t[:, 3].clone(); w_t = uv_t[:, 4].clone()
        uv_t = uv_t[:, :2].clone()
    else:
        if vis is None or weight is None: raise ValueError("Either provide vis & weight, or pass uv with 5 columns.")
        v_t = to_t(vis);  re_t = v_t[:, 0].clone(); im_t = v_t[:, 1].clone(); w_t = to_t(weight).clone()

    H, W = int(img_size[0]), int(img_size[1])
    cell_size_rad = cfg.cell_size_arcsec * (math.pi / 648000.0)

    Cul, Sul, Cvm, Svm = precompute_uv_phases_full(uv_t, H, W, cell_size_rad, dev, torch.float32)
    with torch.no_grad():
        dirty_t, psf_t = make_dirty_and_psf_full(Cul, Sul, Cvm, Svm, re_t, im_t, w_t, normalize=True)

    net = DIPUNet(input_depth=cfg.input_depth, out_ch=1, base_ch=cfg.base_channels, depth=cfg.depth, dropout=cfg.dropout).to(dev)
    z = torch.randn(1, cfg.input_depth, H, W, device=dev, dtype=torch.float32)

    def pos(x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x) if cfg.positivity else x

    student = StudentTLoss(nu=cfg.nu, learn_sigma=cfg.learn_sigma, init_sigma=cfg.init_sigma, reduction="mean").to(dev)
    optimizer = torch.optim.Adam(list(net.parameters()) + ([student.log_sigma] if cfg.learn_sigma else []), lr=cfg.lr)

    
    targ_vis = torch.stack([re_t, im_t], dim=1)
    weights = torch.clamp(w_t, min=1e-12)              # [N]
    
    best_img = None; best_data = float("inf")
    for it in range(cfg.num_iters):
        optimizer.zero_grad(set_to_none=True)
        img = pos(net(z))
        pred_vis = predict_vis_from_precomp(img, Cul, Sul, Cvm, Svm)
        #targ_vis = torch.stack([re_t, im_t], dim=1)
        data_term = student(pred_vis, targ_vis, weight=weights)
        #data_term = student(pred_vis, targ_vis, weight=w_t)
        reg = cfg.tv_weight * tv_loss(img)
        loss = data_term + reg
        loss.backward(); optimizer.step()

        if (it + 1) % cfg.out_every == 0 or it == 0:
            msg = f"[{it+1:5d}/{cfg.num_iters}]  data(full)={float(data_term.detach().cpu()):.6f}  tv={float(reg.detach().cpu()):.6f}"
            if cfg.learn_sigma: msg += f"  sigma={float(torch.exp(student.log_sigma).detach().cpu()):.4g}"
            print(msg)
        with torch.no_grad():
            if data_term.item() < best_data:
                best_data = data_term.item(); best_img = img.detach().clone()

    if best_img is None: best_img = img.detach()
    return best_img[0, 0].detach().cpu().numpy(), dirty_t.detach().cpu().numpy(), psf_t.detach().cpu().numpy()


# =========================================================
# LARGE / MEMORY PATH — stratified sampling + SNIS
# =========================================================
def predict_vis_chunk(img: torch.Tensor, u: torch.Tensor, v: torch.Tensor, l: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    I = img[0, 0]
    ul = 2.0 * math.pi * (u[:, None] * l[None, :])
    vm = 2.0 * math.pi * (v[:, None] * m[None, :])
    Cul, Sul = torch.cos(ul), torch.sin(ul)
    Cvm, Svm = torch.cos(vm), torch.sin(vm)
    A = I @ Cul.t(); B = I @ Sul.t()
    real = torch.sum(Cvm * A.t(), dim=1) - torch.sum(Svm * B.t(), dim=1)
    imag = -(torch.sum(Cvm * B.t(), dim=1) + torch.sum(Svm * A.t(), dim=1))
    return torch.stack([real, imag], dim=1)

@torch.no_grad()
def make_dirty_and_psf_stream(uv: torch.Tensor, re: torch.Tensor, im: torch.Tensor, w: torch.Tensor,
                              H: int, W: int, cell_size_rad: float, device: torch.device,
                              dtype: torch.dtype = torch.float32, normalize: bool = True,
                              chunk_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    if chunk_size is None or chunk_size <= 0: chunk_size = 32768
    l, m = _grid_coords(H, W, cell_size_rad, device, dtype)
    dirty = torch.zeros(H, W, device=device, dtype=dtype)
    psf   = torch.zeros(H, W, device=device, dtype=dtype)
    N = uv.shape[0]
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        u_c = uv[start:end, 0].to(device=device, dtype=dtype)
        v_c = uv[start:end, 1].to(device=device, dtype=dtype)
        re_c = re[start:end].to(device=device, dtype=dtype)
        im_c = im[start:end].to(device=device, dtype=dtype)
        w_c  = w[start:end].to(device=device, dtype=dtype)
        ul = 2.0 * math.pi * (u_c[:, None] * l[None, :]); vm = 2.0 * math.pi * (v_c[:, None] * m[None, :])
        Cul, Sul = torch.cos(ul), torch.sin(ul); Cvm, Svm = torch.cos(vm), torch.sin(vm)
        E = (w_c * re_c)[:, None] * Cul - (w_c * im_c)[:, None] * Sul
        F = (w_c * re_c)[:, None] * Sul + (w_c * im_c)[:, None] * Cul
        dirty += Cvm.t() @ E - Svm.t() @ F
        psf   += Cvm.t() @ (w_c[:, None] * Cul) - Svm.t() @ (w_c[:, None] * Sul)
        del u_c, v_c, re_c, im_c, w_c, ul, vm, Cul, Sul, Cvm, Svm, E, F
    if normalize:
        norm = torch.sum(psf).clamp_min(1e-12)
        dirty = dirty / norm; psf = psf / norm
    return dirty, psf


# ===== VisibilitySampler (mixture + SNIS) with large‑N‑safe quantiles + blockwise multinomial =====
class VisibilitySampler:
    def __init__(self, uv: torch.Tensor, w: torch.Tensor, cfg: DIPConfig, device: torch.device):
        self.cfg = cfg; self.dev = device; self.cpu = torch.device("cpu")
        self.N = int(uv.shape[0])
        u_cpu = uv[:, 0].detach().to(self.cpu, dtype=torch.float32)
        v_cpu = uv[:, 1].detach().to(self.cpu, dtype=torch.float32)
        self.rho = torch.sqrt(u_cpu * u_cpu + v_cpu * v_cpu)
        self.w = w.detach().to(self.cpu, dtype=torch.float32).clamp_min(0)
        self._build_radial_bins(); self._build_angular_bins(u_cpu, v_cpu)
        self.uni_prob = torch.tensor(1.0 / float(self.N), dtype=torch.float32, device=self.cpu)
        self._build_inv_radius_prob(); self._build_weight_prob(); self._build_alphas()
        self.bin_loss_ema = torch.zeros(self.B, dtype=torch.float32, device=self.cpu)
        self._make_bin_prob()

    # ---- quantile helpers ----
    def _quantile_edges_full(self, B: int) -> torch.Tensor:
        q = torch.linspace(0, 1, B + 1, dtype=torch.float32, device=self.cpu)
        edges = torch.quantile(self.rho, q); return self._ensure_strictly_increasing(edges)

    def _quantile_edges_sample(self, B: int, k: int) -> torch.Tensor:
        k = int(min(max(1000, k), self.N)); idx = torch.randint(0, self.N, (k,), device=self.cpu)
        sample = self.rho[idx]; q = torch.linspace(0, 1, B + 1, dtype=torch.float32, device=self.cpu)
        edges = torch.quantile(sample, q); return self._ensure_strictly_increasing(edges)

    def _quantile_edges_hist(self, B: int, H: int) -> torch.Tensor:
        rmin = float(self.rho.min().item()); rmax = float(self.rho.max().item()); 
        if rmax <= rmin: rmax = rmin + 1.0
        eps = max(self.cfg.radius_eps_frac * rmax, 1e-12)
        hist = torch.histc(self.rho + eps, bins=int(H), min=rmin, max=rmax + eps)
        cdf = torch.cumsum(hist, dim=0); cdf = cdf / float(cdf[-1].item())
        q = torch.linspace(0, 1, B + 1, dtype=torch.float32)
        edges = torch.zeros(B + 1, dtype=torch.float32); j = 0
        for i in range(B + 1):
            while j < H - 1 and cdf[j] < q[i]: j += 1
            edges[i] = rmin + (rmax - rmin) * (j / max(1, H - 1))
        return self._ensure_strictly_increasing(edges)

    @staticmethod
    def _ensure_strictly_increasing(edges: torch.Tensor) -> torch.Tensor:
        edges = edges.clone(); span = float(edges[-1].item() - edges[0].item() + 1.0); tiny = 1e-9 * span
        for i in range(1, edges.numel()):
            if edges[i] <= edges[i - 1]: edges[i] = edges[i - 1] + tiny
        return edges

    def _build_radial_bins(self):
        B = max(1, int(self.cfg.radial_bins))
        if not self.cfg.radial_quantiles or self.cfg.quantile_mode.lower() == "logspace":
            rmin = float(self.rho.min().item()); rmax = float(self.rho.max().item()); 
            if rmax <= rmin: rmax = rmin + 1.0
            eps = max(self.cfg.radius_eps_frac * rmax, 1e-12)
            lo = math.log10(max(rmin + eps, 1e-12)); hi = math.log10(rmax + eps)
            edges = torch.logspace(lo, hi, steps=B + 1, base=10.0, dtype=torch.float32) - eps
            edges = self._ensure_strictly_increasing(edges)
        else:
            mode = self.cfg.quantile_mode.lower(); N = self.N
            if mode == "auto":
                if N <= self.cfg.quantile_full_threshold: edges = self._quantile_edges_full(B)
                elif N <= max(self.cfg.quantile_sample_max, 1_000_000): edges = self._quantile_edges_sample(B, self.cfg.quantile_sample_max)
                else: edges = self._quantile_edges_hist(B, self.cfg.hist_bins_for_quantiles)
            elif mode == "full": edges = self._quantile_edges_full(B)
            elif mode == "sample": edges = self._quantile_edges_sample(B, self.cfg.quantile_sample_max)
            elif mode == "hist": edges = self._quantile_edges_hist(B, self.cfg.hist_bins_for_quantiles)
            else:
                rmin = float(self.rho.min().item()); rmax = float(self.rho.max().item()); 
                if rmax <= rmin: rmax = rmin + 1.0
                edges = torch.linspace(rmin, rmax, steps=B + 1, dtype=torch.float32)
                edges = self._ensure_strictly_increasing(edges)
        self.edges = edges.to(self.cpu)
        bin_idx = torch.bucketize(self.rho, self.edges, right=False) - 1
        bin_idx = torch.clamp(bin_idx, 0, B - 1).to(torch.long)
        self.bin_idx = bin_idx
        self.B = B
        self.bin_indices: List[torch.Tensor] = []
        self.bin_counts = torch.zeros(B, dtype=torch.long)
        for b in range(B):
            idx_b = torch.where(self.bin_idx == b)[0]; self.bin_indices.append(idx_b); self.bin_counts[b] = idx_b.numel()
        self.nonempty_bins = torch.tensor([b for b in range(B) if self.bin_counts[b] > 0], dtype=torch.long)
        if self.nonempty_bins.numel() == 0: self.nonempty_bins = torch.tensor([0], dtype=torch.long)

    def _build_angular_bins(self, u_cpu: torch.Tensor, v_cpu: torch.Tensor):
        A = int(self.cfg.angular_bins)
        if A <= 0:
            self.A = 0; self.theta_idx = None; self.ang_counts = None; self.ang_indices = None; return
        theta = torch.atan2(v_cpu, u_cpu); theta = (theta + math.pi) / (2.0 * math.pi)
        edges = torch.linspace(0.0, 1.0, steps=A + 1)
        t_idx = torch.bucketize(theta, edges, right=False) - 1
        t_idx = torch.clamp(t_idx, 0, A - 1).to(torch.long)
        self.theta_idx = t_idx; self.A = A
        self.ang_counts = torch.zeros(A, dtype=torch.long); self.ang_indices: List[torch.Tensor] = []
        for a in range(A):
            idx_a = torch.where(self.theta_idx == a)[0]; self.ang_indices.append(idx_a); self.ang_counts[a] = idx_a.numel()

    def _build_inv_radius_prob(self):
        r = self.rho; rmax = float(r.max().item()) if self.N > 0 else 1.0
        eps = max(self.cfg.radius_eps_frac * rmax, 1e-12); gamma = float(self.cfg.inv_radius_gamma)
        x = 1.0 / torch.pow(r + eps, gamma); total = float(torch.sum(x).item())
        if total <= 0: self.p_inv = torch.full((self.N,), 1.0 / self.N, dtype=torch.float32, device=self.cpu)
        else: self.p_inv = (x / total).to(torch.float32)

    def _build_weight_prob(self):
        w = self.w; total = float(torch.sum(w).item())
        if total <= 0: self.p_w = torch.full((self.N,), 1.0 / self.N, dtype=torch.float32, device=self.cpu)
        else: self.p_w = (w / total).to(torch.float32)

    def _build_alphas(self):
        alphas = torch.tensor([
            max(0.0, float(self.cfg.alpha_uniform)),
            max(0.0, float(self.cfg.alpha_radial)),
            max(0.0, float(self.cfg.alpha_inv_radius)),
            max(0.0, float(self.cfg.alpha_weight)),
            max(0.0, float(self.cfg.alpha_angular)) if self.cfg.angular_bins > 0 else 0.0,
        ], dtype=torch.float32, device=self.cpu)
        s = float(alphas.sum().item()); 
        if s <= 0: alphas = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.cpu)
        else: alphas = alphas / s
        self.alphas = alphas

    def _make_bin_prob(self):
        B = self.B; prob = torch.zeros(B, dtype=torch.float32, device=self.cpu)
        if self.nonempty_bins.numel() > 0: prob[self.nonempty_bins] = 1.0 / float(self.nonempty_bins.numel())
        K = int(self.cfg.low_k_bins); K = max(0, min(K, B))
        if K > 0 and self.nonempty_bins.numel() > 0:
            boost = max(1.0, float(self.cfg.low_k_boost))
            low_bins = torch.arange(0, K, dtype=torch.long); low_bins = low_bins[torch.isin(low_bins, self.nonempty_bins)]
            prob[low_bins] = prob[low_bins] * boost
        if int(self.cfg.adapt_every) > 0:
            ema = torch.clamp(self.bin_loss_ema, min=0.0)
            if torch.any(ema > 0):
                ema = ema ** float(self.cfg.adapt_power)
                ema_masked = torch.zeros_like(prob); ema_masked[self.nonempty_bins] = ema[self.nonempty_bins]
                if float(ema_masked.sum().item()) > 0: prob = ema_masked
        s = float(prob[self.nonempty_bins].sum().item())
        if s <= 0: prob[self.nonempty_bins] = 1.0 / float(self.nonempty_bins.numel())
        else: prob[self.nonempty_bins] = prob[self.nonempty_bins] / s
        self.bin_prob = prob

    @staticmethod
    def _multinomial_large(p: torch.Tensor, num_samples: int, replacement: bool = True,
                           max_cats_safe: int = (1 << 24) - 4096, block_size: int = 8_000_000) -> torch.Tensor:
        N = int(p.numel())
        if N <= max_cats_safe:
            if float(p.sum().item()) <= 0: return torch.randint(0, N, (num_samples,), device=p.device)
            return torch.multinomial(p, num_samples, replacement=replacement)
        B = max(1, (N + block_size - 1) // block_size)
        block_weights = torch.empty(B, dtype=torch.float32, device=p.device)
        for b in range(B):
            s = b * block_size; e = min(s + block_size, N); block_weights[b] = p[s:e].sum()
        total_bw = float(block_weights.sum().item())
        if total_bw <= 0: return torch.randint(0, N, (num_samples,), device=p.device)
        block_probs = block_weights / total_bw
        block_ids = torch.multinomial(block_probs, num_samples, replacement=True)
        counts = torch.bincount(block_ids, minlength=B)
        out_parts = []
        for b in range(B):
            k = int(counts[b].item()); 
            if k == 0: continue
            s = b * block_size; e = min(s + block_size, N)
            pb = p[s:e]; ps = float(pb.sum().item())
            if ps <= 0: idx_local = torch.randint(0, e - s, (k,), device=p.device)
            else: idx_local = torch.multinomial(pb / ps, k, replacement=True)
            out_parts.append(s + idx_local)
        out = torch.cat(out_parts, dim=0); perm = torch.randperm(out.numel(), device=out.device)
        return out[perm]

    @torch.no_grad()
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = int(batch_size); 
        if bs <= 0: raise ValueError("batch_size must be > 0")
        comp_idx = torch.multinomial(self.alphas, bs, replacement=True)
        counts = torch.bincount(comp_idx, minlength=5); nU, nR, nI, nW, nA = [int(counts[i].item()) for i in range(5)]
        idx_list: List[torch.Tensor] = []

        if nU > 0:
            iu = torch.randint(0, self.N, (nU,), device=self.cpu); idx_list.append(iu)

        if nR > 0:
            prob_ne = self.bin_prob[self.nonempty_bins]; prob_ne = prob_ne / float(prob_ne.sum().item())
            bsel = torch.multinomial(prob_ne, nR, replacement=True); bins = self.nonempty_bins[bsel]
            uniq_bins, per_counts = bins.unique(return_counts=True); ir_parts = []
            for b, k in zip(uniq_bins.tolist(), per_counts.tolist()):
                nb_count = int(self.bin_counts[int(b)].item())
                if nb_count <= 0: ir_parts.append(torch.randint(0, self.N, (k,), device=self.cpu))
                else:
                    pos = torch.randint(0, nb_count, (k,), device=self.cpu)
                    ir_parts.append(self.bin_indices[int(b)][pos])
            ir = torch.cat(ir_parts, dim=0).view(-1); idx_list.append(ir)

        if nI > 0:
            ii = self._multinomial_large(self.p_inv, nI, replacement=True); idx_list.append(ii)

        if nW > 0:
            if float(self.p_w.sum().item()) <= 0: iw = torch.randint(0, self.N, (nW,), device=self.cpu)
            else: iw = self._multinomial_large(self.p_w, nW, replacement=True)
            idx_list.append(iw)

        if nA > 0 and hasattr(self, "ang_indices") and self.ang_indices is not None:
            nonempty_ang = torch.tensor([a for a in range(self.A) if self.ang_counts[a] > 0], dtype=torch.long)
            if nonempty_ang.numel() == 0: ia = torch.randint(0, self.N, (nA,), device=self.cpu)
            else:
                pang = torch.full((nonempty_ang.numel(),), 1.0 / float(nonempty_ang.numel()), dtype=torch.float32)
                asel = torch.multinomial(pang, nA, replacement=True); bins_a = nonempty_ang[asel]
                ia_parts = []
                for a in bins_a.tolist():
                    arr = self.ang_indices[int(a)]; n_a = int(arr.numel()); pos = torch.randint(0, n_a, (1,), device=self.cpu)
                    ia_parts.append(arr[pos])
                ia = torch.cat(ia_parts, dim=0).view(-1)
            idx_list.append(ia)

        idx_cpu = torch.cat(idx_list, dim=0).to(torch.long)

        b_i = self.bin_idx[idx_cpu]; n_b = self.bin_counts[b_i].to(torch.float32)
        p_rad = torch.where(n_b > 0, self.bin_prob[b_i].to(torch.float32) * (1.0 / n_b), torch.zeros_like(n_b))
        p_inv = self.p_inv[idx_cpu]; p_w = self.p_w[idx_cpu]; p_ang = torch.zeros_like(n_b)

        αU, αR, αI, αW, αA = [self.alphas[i].item() for i in range(5)]
        p_mix = (αU * float(self.uni_prob.item()) + αR * p_rad + αI * p_inv + αW * p_w + αA * p_ang).clamp_min(1e-18)

        perm = torch.randperm(idx_cpu.numel()); idx_cpu = idx_cpu[perm]; p_mix = p_mix[perm]
        return idx_cpu.to(self.dev), p_mix.to(self.dev)


def reconstruct_dip_memory(
    uv: np.ndarray | torch.Tensor,
    vis: Optional[np.ndarray | torch.Tensor] = None,
    weight: Optional[np.ndarray | torch.Tensor] = None,
    img_size: Tuple[int, int] = (256, 256),
    cfg: Optional[DIPConfig] = None,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cfg is None: cfg = DIPConfig()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    dev = pick_device(device)

    def to_t(x):
        if x is None: return None
        if torch.is_tensor(x): return x.to(dev, dtype=torch.float32)
        return torch.as_tensor(x, device=dev, dtype=torch.float32)

    uv_t = to_t(uv)
    if uv_t.ndim != 2: raise ValueError("uv must be 2D")
    if vis is None and weight is None and uv_t.shape[1] >= 5:
        re_t = uv_t[:, 2].clone(); im_t = uv_t[:, 3].clone(); w_t = uv_t[:, 4].clone(); uv_t = uv_t[:, :2].clone()
    else:
        if vis is None or weight is None: raise ValueError("Either provide vis & weight, or pass uv with 5 columns.")
        v_t = to_t(vis); re_t = v_t[:, 0].clone(); im_t = v_t[:, 1].clone(); w_t = to_t(weight).clone()

    H, W = int(img_size[0]), int(img_size[1])
    cell_size_rad = cfg.cell_size_arcsec * (math.pi / 648000.0)
    l, m = _grid_coords(H, W, cell_size_rad, dev, torch.float32)

    stream_chunk = cfg.stream_chunk_vis if (cfg.stream_chunk_vis and cfg.stream_chunk_vis > 0) else cfg.batch_vis
    with torch.no_grad():
        dirty_t, psf_t = make_dirty_and_psf_stream(uv_t, re_t, im_t, w_t, H, W, cell_size_rad, dev, torch.float32, True, stream_chunk)

    net = DIPUNet(input_depth=cfg.input_depth, out_ch=1, base_ch=cfg.base_channels, depth=cfg.depth, dropout=cfg.dropout).to(dev)
    z = torch.randn(1, cfg.input_depth, H, W, device=dev, dtype=torch.float32)

    def pos(x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x) if cfg.positivity else x

    student = StudentTLoss(nu=cfg.nu, learn_sigma=cfg.learn_sigma, init_sigma=cfg.init_sigma, reduction="none").to(dev)
    optimizer = torch.optim.Adam(list(net.parameters()) + ([student.log_sigma] if cfg.learn_sigma else []), lr=cfg.lr)

    sampler = VisibilitySampler(uv_t, w_t, cfg, dev)
    N = uv_t.shape[0]; bs = max(1, min(cfg.batch_vis, N))
    u_unif = 1.0 / float(N)

    targ_b = torch.stack([re_b, im_b], dim=1)
    weights_b = torch.clamp(w_b, min=1e-12)              # [N]

    best_img = None; best_data = float("inf")
    for it in range(cfg.num_iters):
        optimizer.zero_grad(set_to_none=True)
        img = pos(net(z))

        idx, p_mix = sampler.sample(bs); p_mix = torch.clamp(p_mix, min=1e-12)
        u_b = uv_t[idx, 0]; v_b = uv_t[idx, 1]; re_b = re_t[idx]; im_b = im_t[idx]; w_b = w_t[idx]

        pred_b = predict_vis_chunk(img, u_b, v_b, l, m)
        #targ_b = torch.stack([re_b, im_b], dim=1)
        #nll_vec = student(pred_b, targ_b, weight=w_b)
        nll_vec = student(pred_b, targ_b, weight=weights_b)

        if cfg.importance_snis:
            imp = (u_unif / p_mix); data_term = torch.sum(imp * nll_vec) / (torch.sum(imp) + 1e-12)
        else:
            data_term = torch.mean(nll_vec * (u_unif / p_mix))

        reg = cfg.tv_weight * tv_loss(img); loss = data_term + reg
        loss.backward(); optimizer.step()

        if cfg.adapt_every > 0 and ((it + 1) % cfg.adapt_every == 0):
            # Optional adaptive bin weighting (uses per-bin mean NLL)
            sampler.update_with_batch(idx, nll_vec)

        if (it + 1) % cfg.out_every == 0 or it == 0:
            msg = f"[{it+1:5d}/{cfg.num_iters}]  data(strat)={float(data_term.detach().cpu()):.6f}  tv={float(reg.detach().cpu()):.6f}"
            if cfg.learn_sigma: msg += f"  sigma={float(torch.exp(student.log_sigma).detach().cpu()):.4g}"
            print(msg)
        with torch.no_grad():
            if data_term.item() < best_data:
                best_data = data_term.item(); best_img = img.detach().clone()

    if best_img is None: best_img = img.detach()
    return best_img[0, 0].detach().cpu().numpy(), dirty_t.detach().cpu().numpy(), psf_t.detach().cpu().numpy()


# =========================================================
# AUTO SELECTOR
# =========================================================
def reconstruct_dip(
    uv: np.ndarray | torch.Tensor,
    vis: Optional[np.ndarray | torch.Tensor] = None,
    weight: Optional[np.ndarray | torch.Tensor] = None,
    img_size: Tuple[int, int] = (256, 256),
    cfg: Optional[DIPConfig] = None,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cfg is None: cfg = DIPConfig()
    if torch.is_tensor(uv): uv_shape = uv.shape
    else: uv_shape = np.asarray(uv).shape
    if len(uv_shape) != 2: raise ValueError("uv must have shape [N,2] or [N,5]")
    N = uv_shape[0]
    mode = cfg.force_mode.lower()
    if mode not in ("auto", "small", "memory"): raise ValueError("cfg.force_mode must be 'auto' | 'small' | 'memory'")
    use_small = (mode == "small") or (mode == "auto" and N <= int(cfg.auto_threshold_n))
    if use_small:
        print(f"[selector] Using SMALL pipeline (N={N:,} ≤ threshold {cfg.auto_threshold_n:,}).")
        return reconstruct_dip_small(uv, vis=vis, weight=weight, img_size=img_size, cfg=cfg, device=device)
    else:
        print(f"[selector] Using MEMORY pipeline (N={N:,} > threshold {cfg.auto_threshold_n:,}).")
        return reconstruct_dip_memory(uv, vis=vis, weight=weight, img_size=img_size, cfg=cfg, device=device)


# =========================================================
# BOOTSTRAP: per-pixel flux uncertainty (std) via Poisson/Bayesian bootstrap
# =========================================================

# ---- deterministic hash-based RNG over indices (CPU; no N-length vectors) ----
def _hashed_uniform_01(idx_cpu: np.ndarray, replicate: int, seed: int = 0) -> np.ndarray:
    """
    Stateles uniform(0,1) from index+replicate using SplitMix64 mixing.
    idx_cpu: np.uint64 array of indices
    """
    x = (idx_cpu.astype(np.uint64) ^
         np.uint64(seed) ^
         (np.uint64(replicate) * np.uint64(0x9E3779B97F4A7C15)))
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x ^= (x >> np.uint64(30)); x = (x * np.uint64(0xbf58476d1ce4e5b9)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x ^= (x >> np.uint64(27)); x = (x * np.uint64(0x94d049bb133111eb)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x ^= (x >> np.uint64(31))
    # Map to [0,1): take top 53 bits as mantissa
    u = ((x >> np.uint64(11)).astype(np.float64)) * (1.0 / (1 << 53))
    return u  # float64 in [0,1)

def _bootstrap_weights_for_indices(idx: torch.Tensor, method: str, replicate: int, seed: int,
                                   device: torch.device) -> torch.Tensor:
    """
    Returns bootstrap weight s_i for each index in idx.
    method: 'poisson' (counts), 'bayesian' (Exp(1) weights).
    """
    idx_np = idx.detach().to("cpu").numpy().astype(np.uint64)
    u = _hashed_uniform_01(idx_np, replicate, seed)
    if method == "poisson":
        # Inverse CDF for Poisson(1) via fixed thresholds
        # CDF: k=0..8  (prob of >8 is tiny)
        cdf = np.array([0.36787944117144233, 0.7357588823428847, 0.9196986029286059,
                        0.9810118431238463, 0.9963401531726563, 0.9994058151824183,
                        0.9999167588507110, 0.9999897508033256, 0.9999988742690519], dtype=np.float64)
        s = np.zeros_like(u, dtype=np.float32)
        s += (u >= cdf[0]).astype(np.float32)
        s += (u >= cdf[1]).astype(np.float32)
        s += (u >= cdf[2]).astype(np.float32)
        s += (u >= cdf[3]).astype(np.float32)
        s += (u >= cdf[4]).astype(np.float32)
        s += (u >= cdf[5]).astype(np.float32)
        s += (u >= cdf[6]).astype(np.float32)
        s += (u >= cdf[7]).astype(np.float32)
        s += (u >= cdf[8]).astype(np.float32)  # very rare >8 events counted as 9 here
        # s is approx Poisson(1) count
        return torch.as_tensor(s, device=device, dtype=torch.float32)
    elif method == "bayesian":
        # Bayesian bootstrap: weights ~ Exp(1)
        s = -np.log(np.clip(u, 1e-16, 1.0))
        return torch.as_tensor(s, device=device, dtype=torch.float32)
    else:
        raise ValueError("bootstrap method must be 'poisson' or 'bayesian'")


def _train_small_bootstrap_once(uv_t, re_t, im_t, w_t, img_size, cfg: DIPConfig, device, replicate: int,
                                boot_method: str, boot_seed: int) -> np.ndarray:
    """Single bootstrap realization using SMALL path with per-sample weights."""
    torch.manual_seed(cfg.seed + replicate); np.random.seed(cfg.seed + replicate)
    dev = pick_device(device)
    H, W = int(img_size[0]), int(img_size[1])
    cell_size_rad = cfg.cell_size_arcsec * (math.pi / 648000.0)

    Cul, Sul, Cvm, Svm = precompute_uv_phases_full(uv_t, H, W, cell_size_rad, dev, torch.float32)

    net = DIPUNet(input_depth=cfg.input_depth, out_ch=1, base_ch=cfg.base_channels, depth=cfg.depth, dropout=cfg.dropout).to(dev)
    z = torch.randn(1, cfg.input_depth, H, W, device=dev, dtype=torch.float32)
    def pos(x): return F.softplus(x) if cfg.positivity else x
    student = StudentTLoss(nu=cfg.nu, learn_sigma=cfg.learn_sigma, init_sigma=cfg.init_sigma, reduction="none").to(dev)
    optimizer = torch.optim.Adam(list(net.parameters()) + ([student.log_sigma] if cfg.learn_sigma else []), lr=cfg.lr)

    N = uv_t.shape[0]
    all_idx = torch.arange(N, device=dev, dtype=torch.long)

    best_img = None; best_data = float("inf")
    for it in range(cfg.num_iters):
        optimizer.zero_grad(set_to_none=True)
        img = pos(net(z))
        pred_vis = predict_vis_from_precomp(img, Cul, Sul, Cvm, Svm)     # [N,2]
        targ_vis = torch.stack([re_t, im_t], dim=1)                      # [N,2]
        nll_vec = student(pred_vis, targ_vis, weight=w_t)                # [N]

        # Bootstrap weights for all indices (computed deterministically via hash)
        s = _bootstrap_weights_for_indices(all_idx, boot_method, replicate, cfg.seed + boot_seed, dev)  # [N]
        num = torch.sum(s * nll_vec); denom = torch.sum(s) + 1e-12
        data_term = num / denom

        reg = cfg.tv_weight * tv_loss(img); loss = data_term + reg
        loss.backward(); optimizer.step()

        if (it + 1) % cfg.out_every == 0 or it == 0:
            print(f"[boot {replicate:03d}] [{it+1:5d}/{cfg.num_iters}] data(full*boot)={float(data_term.detach().cpu()):.6f} tv={float(reg.detach().cpu()):.6f}")

        with torch.no_grad():
            if data_term.item() < best_data:
                best_data = data_term.item(); best_img = img.detach().clone()

    if best_img is None: best_img = img.detach()
    return best_img[0, 0].detach().cpu().numpy()


def _train_memory_bootstrap_once(uv_t, re_t, im_t, w_t, img_size, cfg: DIPConfig, device, replicate: int,
                                 boot_method: str, boot_seed: int) -> np.ndarray:
    """Single bootstrap realization using MEMORY path with SNIS + bootstrap multipliers."""
    torch.manual_seed(cfg.seed + replicate); np.random.seed(cfg.seed + replicate)
    dev = pick_device(device)
    H, W = int(img_size[0]), int(img_size[1])
    cell_size_rad = cfg.cell_size_arcsec * (math.pi / 648000.0)
    l, m = _grid_coords(H, W, cell_size_rad, dev, torch.float32)

    # No need to rebuild dirty/psf for each replicate (not used in optimization)
    net = DIPUNet(input_depth=cfg.input_depth, out_ch=1, base_ch=cfg.base_channels, depth=cfg.depth, dropout=cfg.dropout).to(dev)
    z = torch.randn(1, cfg.input_depth, H, W, device=dev, dtype=torch.float32)
    def pos(x): return F.softplus(x) if cfg.positivity else x

    student = StudentTLoss(nu=cfg.nu, learn_sigma=cfg.learn_sigma, init_sigma=cfg.init_sigma, reduction="none").to(dev)
    optimizer = torch.optim.Adam(list(net.parameters()) + ([student.log_sigma] if cfg.learn_sigma else []), lr=cfg.lr)

    sampler = VisibilitySampler(uv_t, w_t, cfg, dev)
    N = uv_t.shape[0]; bs = max(1, min(cfg.batch_vis, N))
    u_unif = 1.0 / float(N)

    best_img = None; best_data = float("inf")
    for it in range(cfg.num_iters):
        optimizer.zero_grad(set_to_none=True)
        img = pos(net(z))

        idx, p_mix = sampler.sample(bs); p_mix = torch.clamp(p_mix, min=1e-12)
        u_b = uv_t[idx, 0]; v_b = uv_t[idx, 1]; re_b = re_t[idx]; im_b = im_t[idx]; w_b = w_t[idx]

        pred_b = predict_vis_chunk(img, u_b, v_b, l, m)
        targ_b = torch.stack([re_b, im_b], dim=1)
        nll_vec = student(pred_b, targ_b, weight=w_b)  # [bs]

        # Bootstrap weights s_i for this batch (deterministic per index/replicate)
        s_i = _bootstrap_weights_for_indices(idx, boot_method, replicate, cfg.seed + boot_seed, dev)  # [bs]

        if cfg.importance_snis:
            imp = (u_unif / p_mix)
            num = torch.sum(imp * s_i * nll_vec); denom = torch.sum(imp * s_i) + 1e-12
            data_term = num / denom
        else:
            data_term = torch.mean(nll_vec * (u_unif / p_mix) * s_i)

        reg = cfg.tv_weight * tv_loss(img); loss = data_term + reg
        loss.backward(); optimizer.step()

        if cfg.adapt_every > 0 and ((it + 1) % cfg.adapt_every == 0):
            sampler.update_with_batch(idx, nll_vec)

        if (it + 1) % cfg.out_every == 0 or it == 0:
            print(f"[boot {replicate:03d}] [{it+1:5d}/{cfg.num_iters}] data(strat*boot)={float(data_term.detach().cpu()):.6f} tv={float(reg.detach().cpu()):.6f}")

        with torch.no_grad():
            if data_term.item() < best_data:
                best_data = data_term.item(); best_img = img.detach().clone()

    if best_img is None: best_img = img.detach()
    return best_img[0, 0].detach().cpu().numpy()


def bootstrap_reconstruct(
    uv: np.ndarray | torch.Tensor,
    vis: Optional[np.ndarray | torch.Tensor],
    weight: Optional[np.ndarray | torch.Tensor],
    img_size: Tuple[int, int],
    cfg: DIPConfig,
    device: Optional[str] = None,
    *,
    B: int = 20,
    method: str = "poisson",          # "poisson" or "bayesian"
    percentiles: Tuple[float, float] = (16.0, 84.0),
    return_all: bool = False
) -> Dict[str, np.ndarray]:
    """
    Run B bootstrap replicas and return per-pixel mean/std and (p_lo, p_hi).
    - method='poisson': classical Poisson bootstrap (counts via hash-based RNG)
    - method='bayesian': Bayesian bootstrap (Exp(1) weights)
    """
    if cfg is None:
        cfg = DIPConfig()
    dev = pick_device(device)

    # Move inputs to device once
    def to_t(x):
        if x is None: return None
        if torch.is_tensor(x): return x.to(dev, dtype=torch.float32)
        return torch.as_tensor(x, device=dev, dtype=torch.float32)

    uv_t = to_t(uv)
    if uv_t.ndim != 2: raise ValueError("uv must be 2D")
    if vis is None and weight is None and uv_t.shape[1] >= 5:
        re_t = uv_t[:, 2].clone(); im_t = uv_t[:, 3].clone(); w_t = uv_t[:, 4].clone(); uv_t = uv_t[:, :2].clone()
    else:
        if vis is None or weight is None: raise ValueError("Either provide vis & weight, or pass uv with 5 columns.")
        v_t = to_t(vis); re_t = v_t[:, 0].clone(); im_t = v_t[:, 1].clone(); w_t = to_t(weight).clone()

    # Decide which path to use (same rule as reconstruct_dip)
    N = int(uv_t.shape[0])
    mode = cfg.force_mode.lower()
    use_small = (mode == "small") or (mode == "auto" and N <= int(cfg.auto_threshold_n))

    H, W = int(img_size[0]), int(img_size[1])
    imgs = np.empty((B, H, W), dtype=np.float32)
    boot_seed = 12345  # separate stream from training seed

    for b in range(B):
        if use_small:
            img_b = _train_small_bootstrap_once(uv_t, re_t, im_t, w_t, img_size, cfg, dev, b, method, boot_seed)
        else:
            img_b = _train_memory_bootstrap_once(uv_t, re_t, im_t, w_t, img_size, cfg, dev, b, method, boot_seed)
        imgs[b] = img_b

    mean = imgs.mean(axis=0)
    std  = imgs.std(axis=0, ddof=1) if B > 1 else np.zeros_like(mean)
    p_lo, p_hi = np.percentile(imgs, percentiles[0], axis=0), np.percentile(imgs, percentiles[1], axis=0)

    out = {"mean": mean, "std": std, "p_lo": p_lo, "p_hi": p_hi}
    if return_all: out["samples"] = imgs
    return out
