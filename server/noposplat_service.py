import logging
import sys
import torch as tch
import torchvision.transforms as tf
import io
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path
from plyfile import PlyData, PlyElement
from typing import List, Optional, Tuple, Dict
from PIL import Image
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

_current_dir = Path(__file__).parent
NOPOSPLAT_ROOT = _current_dir / "NoPoSplat"
SRC_ROOT = NOPOSPLAT_ROOT / "src"
if str(NOPOSPLAT_ROOT) not in sys.path:
    sys.path.insert(0, str(NOPOSPLAT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

logger = logging.getLogger("uvicorn")

class CUDADependencyMissingError(RuntimeError):
    pass

_CUDA_DEPENDENCIES_AVAILABLE: bool = True
_model: Optional["ModelWrapper"] = None
_config: Optional["RootCfg"] = None

try:
    from src.config import load_typed_root_config, RootCfg
    from src.global_cfg import set_cfg
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper
    from src.loss import get_losses
    from src.misc.step_tracker import StepTracker
    from src.model.types import Gaussians as NoPoSplatGaussians
    from src.dataset.types import BatchedViews
    from src.dataset.shims.crop_shim import apply_crop_shim_to_views
    from src.dataset.shims.normalize_shim import normalize_image
except ImportError as ie:
    if "diff_gaussian_rasterization" in str(ie).lower():
        logger.warning(f"Failed to import NoPoSplat CUDA components: {ie}")
        _CUDA_DEPENDENCIES_AVAILABLE = False
        ModelWrapper = None
        RootCfg = None
    else:
        raise

_device_str: str = "cpu"
_target_image_shape: Tuple[int, int] = (256, 256)
_default_fov_degrees: float = 60.0

def get_tch_device_str() -> str:
    return "cuda" if tch.cuda.is_available() else "cpu"

def init_noposplat_model():
    global _model, _config, _device_str, _target_image_shape, _CUDA_DEPENDENCIES_AVAILABLE
    
    if _model is not None:
        return
    
    _device_str = get_tch_device_str()
    
    if _device_str == "cpu" or not _CUDA_DEPENDENCIES_AVAILABLE:
        _model = None
        _CUDA_DEPENDENCIES_AVAILABLE = False
        return
    
    try:
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(NOPOSPLAT_ROOT / "config"), version_base=None):
            cfg = compose(config_name="main", overrides=["+experiment=re10k"])
        
        cfg.mode = "test"
        cfg.wandb.mode = "disabled"
        checkpoint_file = NOPOSPLAT_ROOT / "pretrained_weights" / "re10k.ckpt"
        cfg.checkpointing.load = str(checkpoint_file)
        
        if hasattr(cfg, "hydra"):
            OmegaConf.set_struct(cfg, False)
            del cfg.hydra
            OmegaConf.set_struct(cfg, True)
        
        final_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        set_cfg(final_cfg)
        typed_cfg = load_typed_root_config(final_cfg)
        _config = typed_cfg
        
        if hasattr(_config.dataset, "input_image_shape") and _config.dataset.input_image_shape:
            _target_image_shape = tuple(_config.dataset.input_image_shape)
        
        model_instance = ModelWrapper(
            optimizer_cfg=typed_cfg.optimizer,
            test_cfg=typed_cfg.test,
            train_cfg=typed_cfg.train,
            encoder=get_encoder(typed_cfg.model.encoder)[0],
            encoder_visualizer=None,
            decoder=get_decoder(typed_cfg.model.decoder),
            losses=get_losses(typed_cfg.loss),
            step_tracker=StepTracker(),
            distiller=None,
        )
        
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
        
        ckpt_data = tch.load(checkpoint_file, map_location=tch.device(_device_str))
        model_instance.load_state_dict(ckpt_data.get("state_dict", ckpt_data))
        model_instance.to(_device_str)
        model_instance.eval()
        _model = model_instance
        
    except Exception:
        _model = None
        _CUDA_DEPENDENCIES_AVAILABLE = False
        raise

def _calc_intrinsics(w: int, h: int, fov: float) -> tch.Tensor:
    f = w / (2 * np.tan(np.deg2rad(fov / 2)))
    return tch.tensor([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=tch.float32)

def _prepare_context(images: List[Image.Image]) -> Dict[str, tch.Tensor]:
    img_tensors = []
    intrinsics_tensors = []
    
    for img in images:
        w, h = img.size
        img_tensors.append(tf.ToTensor()(img))
        intrinsics_tensors.append(_calc_intrinsics(w, h, _default_fov_degrees))
    
    batched_images = tch.stack(img_tensors).unsqueeze(0).to(_device_str)
    batched_intrinsics = tch.stack(intrinsics_tensors).unsqueeze(0).to(_device_str)
    
    initial_views = {"image": batched_images, "intrinsics": batched_intrinsics}
    cropped_views = apply_crop_shim_to_views(initial_views, _target_image_shape)
    normalized_images = normalize_image(cropped_views["image"], mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    
    num_views = len(images)
    identity_extrinsics = tch.eye(4, dtype=tch.float32, device=_device_str)
    batched_extrinsics = identity_extrinsics.unsqueeze(0).unsqueeze(0).repeat(1, num_views, 1, 1)
    
    return {
        "image": normalized_images,
        "intrinsics": cropped_views["intrinsics"].to(_device_str),
        "extrinsics": batched_extrinsics,
        "near": tch.full((1, num_views), 0.1, dtype=tch.float32, device=_device_str),
        "far": tch.full((1, num_views), 100.0, dtype=tch.float32, device=_device_str),
        "index": tch.arange(num_views, dtype=tch.int64, device=_device_str).unsqueeze(0),
    }

def _extract_gaussians(gaussians: NoPoSplatGaussians) -> Tuple[tch.Tensor, tch.Tensor, tch.Tensor, tch.Tensor, tch.Tensor]:
    means = gaussians.means[0]
    covariances = gaussians.covariances[0]
    harmonics = gaussians.harmonics[0]
    opacities = gaussians.opacities[0]
    
    U, S, Vh = tch.linalg.svd(covariances)
    scales = tch.sqrt(tch.clamp(S, min=1e-8))
    
    R = U @ Vh
    det = tch.det(R)
    R = tch.where(det[:, None, None] < 0, -R, R)
    
    w = tch.sqrt(tch.clamp(1 + R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2], min=1e-8)) / 2
    x = (R[..., 2, 1] - R[..., 1, 2]) / (4 * w)
    y = (R[..., 0, 2] - R[..., 2, 0]) / (4 * w)
    z = (R[..., 1, 0] - R[..., 0, 1]) / (4 * w)
    rotations = tch.stack([x, y, z, w], dim=-1)
    
    return means, scales, rotations, harmonics, opacities

def _to_ply_bytes(means: tch.Tensor, scales: tch.Tensor, rotations: tch.Tensor, harmonics: tch.Tensor, opacities: tch.Tensor) -> bytes:
    means_np = means.detach().cpu().numpy()
    scales_np = scales.detach().cpu().numpy()
    rots_np = rotations.detach().cpu().numpy()
    harmonics_np = harmonics.detach().cpu().numpy()
    opacities_np = opacities.detach().cpu().numpy()
    
    rots_ply = np.stack([rots_np[:, 3], rots_np[:, 0], rots_np[:, 1], rots_np[:, 2]], axis=-1)
    f_dc = harmonics_np[..., 0]
    
    attributes = [
        ("x", means_np[:, 0]), ("y", means_np[:, 1]), ("z", means_np[:, 2]),
        ("nx", np.zeros_like(means_np[:, 0])), ("ny", np.zeros_like(means_np[:, 0])), ("nz", np.zeros_like(means_np[:, 0])),
        ("f_dc_0", f_dc[:, 0]), ("f_dc_1", f_dc[:, 1]), ("f_dc_2", f_dc[:, 2]),
        ("opacity", opacities_np),
        ("scale_0", np.log(np.maximum(scales_np[:, 0], 1e-8))),
        ("scale_1", np.log(np.maximum(scales_np[:, 1], 1e-8))),
        ("scale_2", np.log(np.maximum(scales_np[:, 2], 1e-8))),
        ("rot_0", rots_ply[:, 0]), ("rot_1", rots_ply[:, 1]), ("rot_2", rots_ply[:, 2]), ("rot_3", rots_ply[:, 3]),
    ]
    
    dtype = [(name, "f4") for name, _ in attributes]
    vertices = np.empty(means_np.shape[0], dtype=dtype)
    for i, (name, data) in enumerate(attributes):
        vertices[name] = data
    
    mem_file = io.BytesIO()
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(mem_file)
    return mem_file.getvalue()

def reconstruct_scene_from_images(images: List[Image.Image]) -> bytes:
    if not _CUDA_DEPENDENCIES_AVAILABLE or _model is None:
        raise CUDADependencyMissingError("Reconstruction requires CUDA components")
    if _config is None:
        raise RuntimeError("NoPoSplat service missing config")
    
    context = _prepare_context(images)
    
    with tch.no_grad():
        gaussians = _model.encoder(context, _model.global_step)
        means, scales, rotations, harmonics, opacities = _extract_gaussians(gaussians)
        return _to_ply_bytes(means, scales, rotations, harmonics, opacities)
