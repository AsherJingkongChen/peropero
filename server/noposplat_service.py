import logging
import sys
import torch as tch
import torchvision.transforms as tf
import io
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as SciPyRotation
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
        logger.warning(
            f"Failed to import NoPoSplat CUDA components (e.g., diff_gaussian_rasterization): {ie}. "
            "Reconstruction functionality will be unavailable. This is expected on non-CUDA environments."
        )
        _CUDA_DEPENDENCIES_AVAILABLE = False
        ModelWrapper = None
        RootCfg = None
    else:
        raise

from omegaconf import DictConfig

_device_str: str = "cpu"
_target_image_shape: Tuple[int, int] = (256, 256)
_default_fov_degrees: float = 60.0


def get_tch_device_str() -> str:
    if tch.cuda.is_available():
        return "cuda"
    return "cpu"


def init_noposplat_model():
    global \
        _model, \
        _config, \
        _device_str, \
        _target_image_shape, \
        _CUDA_DEPENDENCIES_AVAILABLE

    if _model is not None:
        logger.info("NoPoSplat model already initialized.")
        return

    _device_str = get_tch_device_str()

    if _device_str == "cpu":
        logger.warning(
            "CUDA is not available. NoPoSplat model initialization requires CUDA and will be skipped. NoPoSplat service will not be available."
        )
        _model = None
        _CUDA_DEPENDENCIES_AVAILABLE = False
        return

    logger.info("Initializing NoPoSplat model and configuration...")
    logger.info(f"NoPoSplat service will use device: {_device_str}")

    if not _CUDA_DEPENDENCIES_AVAILABLE:
        logger.warning(
            "Skipping NoPoSplat model initialization as essential CUDA dependencies are missing (although CUDA device detected)."
        )
        _model = None
        return

    try:
        config_dir = str(NOPOSPLAT_ROOT / "config")
        
        # Clear any existing Hydra instance
        GlobalHydra.instance().clear()
        
        # Use Hydra's compose with overrides to handle all configuration composition
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="main", overrides=["+experiment=re10k"])
        
        # Override specific settings for test mode
        cfg.mode = "test"
        cfg.wandb.mode = "disabled"
        checkpoint_file = NOPOSPLAT_ROOT / "pretrained_weights" / "re10k.ckpt"
        cfg.checkpointing.load = str(checkpoint_file)
        
        # Remove hydra configuration that causes issues
        if hasattr(cfg, "hydra"):
            OmegaConf.set_struct(cfg, False)
            del cfg.hydra
            OmegaConf.set_struct(cfg, True)
        
        # Convert to plain dict and back to ensure clean structure
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        final_cfg = OmegaConf.create(cfg_dict)

        set_cfg(final_cfg)
        typed_cfg = load_typed_root_config(final_cfg)
        _config = typed_cfg

        if (
            hasattr(_config.dataset, "input_image_shape")
            and _config.dataset.input_image_shape
        ):
            _target_image_shape = tuple(_config.dataset.input_image_shape)
        logger.info(f"Target image shape for preprocessing: {_target_image_shape}")

        step_tracker = StepTracker()
        encoder, _ = get_encoder(typed_cfg.model.encoder)
        decoder = get_decoder(typed_cfg.model.decoder)

        current_losses = get_losses(typed_cfg.loss)

        model_instance = ModelWrapper(
            optimizer_cfg=typed_cfg.optimizer,
            test_cfg=typed_cfg.test,
            train_cfg=typed_cfg.train,
            encoder=encoder,
            encoder_visualizer=None,
            decoder=decoder,
            losses=current_losses,
            step_tracker=step_tracker,
            distiller=None,
        )

        if not checkpoint_file.exists():
            _model = None
            _CUDA_DEPENDENCIES_AVAILABLE = False
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

        logger.info(f"Loading checkpoint: {checkpoint_file}")
        ckpt_data = tch.load(checkpoint_file, map_location=tch.device(_device_str))

        state_dict = ckpt_data.get("state_dict", ckpt_data)
        model_instance.load_state_dict(state_dict)

        # Note: global_step is a read-only property in PyTorch Lightning ModelWrapper
        # We skip setting it directly as it causes "can't set attribute" error
        # The model will use its internal global_step management
        logger.info(f"Model loaded. Current global_step: {model_instance.global_step}")

        model_instance.to(_device_str)
        model_instance.eval()
        _model = model_instance
        logger.info("NoPoSplat model initialized successfully.")

    except FileNotFoundError:
        _model = None
        _CUDA_DEPENDENCIES_AVAILABLE = False
        raise
    except Exception:
        _model = None
        _CUDA_DEPENDENCIES_AVAILABLE = False
        raise


def _calculate_initial_intrinsics(
    w_orig: int, h_orig: int, fov_degrees: float
) -> tch.Tensor:
    focal_x = w_orig / (2 * np.tan(np.deg2rad(fov_degrees / 2)))
    focal_y = h_orig / (2 * np.tan(np.deg2rad(fov_degrees / 2)))
    cx_orig = w_orig / 2.0
    cy_orig = h_orig / 2.0
    return tch.tensor(
        [[focal_x, 0, cx_orig], [0, focal_y, cy_orig], [0, 0, 1.0]], dtype=tch.float32
    )


def _gaussians_to_ply_bytes_adapted(
    means: tch.Tensor,
    scales: tch.Tensor,
    rotations_quat_xyzw: tch.Tensor,
    harmonics: tch.Tensor,
    opacities: tch.Tensor,
    save_sh_dc_only: bool = True,
) -> bytes:
    means_np = means.detach().cpu().numpy()
    scales_np = scales.detach().cpu().numpy()
    rotations_np = rotations_quat_xyzw.detach().cpu().numpy()
    harmonics_np = harmonics.detach().cpu().numpy()
    opacities_np = opacities.detach().cpu().numpy()

    x_q, y_q, z_q, w_q = (
        rotations_np[:, 0],
        rotations_np[:, 1],
        rotations_np[:, 2],
        rotations_np[:, 3],
    )
    rotations_for_ply = np.stack((w_q, x_q, y_q, z_q), axis=-1)

    f_dc_np = harmonics_np[..., 0]

    ply_attributes = [
        ("x", means_np[:, 0]),
        ("y", means_np[:, 1]),
        ("z", means_np[:, 2]),
        ("nx", np.zeros_like(means_np[:, 0])),
        ("ny", np.zeros_like(means_np[:, 0])),
        ("nz", np.zeros_like(means_np[:, 0])),
        ("f_dc_0", f_dc_np[:, 0]),
        ("f_dc_1", f_dc_np[:, 1]),
        ("f_dc_2", f_dc_np[:, 2]),
    ]

    if not save_sh_dc_only:
        f_rest_np = harmonics_np[..., 1:].reshape(harmonics_np.shape[0], -1)
        for i in range(f_rest_np.shape[1]):
            ply_attributes.append((f"f_rest_{i}", f_rest_np[:, i]))

    ply_attributes.extend(
        [
            ("opacity", opacities_np),
            ("scale_0", np.log(np.maximum(scales_np[:, 0], 1e-8))),
            ("scale_1", np.log(np.maximum(scales_np[:, 1], 1e-8))),
            ("scale_2", np.log(np.maximum(scales_np[:, 2], 1e-8))),
            ("rot_0", rotations_for_ply[:, 0]),
            ("rot_1", rotations_for_ply[:, 1]),
            ("rot_2", rotations_for_ply[:, 2]),
            ("rot_3", rotations_for_ply[:, 3]),
        ]
    )

    vertex_dtype_list = [(name, "f4") for name, _ in ply_attributes]
    vertices_data_list = [data_col for _, data_col in ply_attributes]

    num_vertices = means_np.shape[0]
    vertices_structured = np.empty(num_vertices, dtype=np.dtype(vertex_dtype_list))
    for i, (name, _) in enumerate(vertex_dtype_list):
        vertices_structured[name] = vertices_data_list[i]

    ply_element = PlyElement.describe(vertices_structured, "vertex")

    mem_file = io.BytesIO()
    PlyData([ply_element], text=False).write(mem_file)
    return mem_file.getvalue()


def reconstruct_scene_from_images(images: List[Image.Image]) -> bytes:
    if not _CUDA_DEPENDENCIES_AVAILABLE or _model is None:
        raise CUDADependencyMissingError(
            "Reconstruction requires CUDA components not available on this system, or the model failed to initialize."
        )

    if _config is None:
        raise RuntimeError(
            "NoPoSplat service is in an inconsistent state (missing config)."
        )

    logger.info(
        f"Processing {len(images)} images for reconstruction on device {_device_str}."
    )

    img_tensors_orig = []
    intrinsics_tensors_orig = []
    for pil_img in images:
        w_orig, h_orig = pil_img.size
        img_tensors_orig.append(tf.ToTensor()(pil_img))
        intrinsics_tensors_orig.append(
            _calculate_initial_intrinsics(w_orig, h_orig, _default_fov_degrees)
        )

    batched_images_orig = tch.stack(img_tensors_orig).unsqueeze(0).to(_device_str)
    batched_intrinsics_orig = (
        tch.stack(intrinsics_tensors_orig).unsqueeze(0).to(_device_str)
    )

    initial_views: BatchedViews = {
        "image": batched_images_orig,
        "intrinsics": batched_intrinsics_orig,
    }

    cropped_views = apply_crop_shim_to_views(initial_views, _target_image_shape)

    normalized_images = normalize_image(
        cropped_views["image"], mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
    )

    num_views = len(images)
    identity_extrinsics = tch.eye(4, dtype=tch.float32, device=_device_str)
    batched_extrinsics = (
        identity_extrinsics.unsqueeze(0).unsqueeze(0).repeat(1, num_views, 1, 1)
    )
    near_planes = tch.full((1, num_views), 0.1, dtype=tch.float32, device=_device_str)
    far_planes = tch.full((1, num_views), 100.0, dtype=tch.float32, device=_device_str)
    indices = tch.arange(num_views, dtype=tch.int64, device=_device_str).unsqueeze(0)

    context_batch: Dict[str, tch.Tensor] = {
        "image": normalized_images,
        "intrinsics": cropped_views["intrinsics"].to(_device_str),
        "extrinsics": batched_extrinsics,
        "near": near_planes,
        "far": far_planes,
        "index": indices,
    }

    global_step_for_encoder = _model.global_step

    with tch.no_grad():
        gaussians_output_obj: NoPoSplatGaussians = _model.encoder(
            context_batch, global_step_for_encoder
        )

        means = gaussians_output_obj.means[0]
        covariances = gaussians_output_obj.covariances[0]
        harmonics = gaussians_output_obj.harmonics[0]
        opacities = gaussians_output_obj.opacities[0]

        num_gaussians = covariances.shape[0]
        all_scales = tch.zeros(num_gaussians, 3, device=_device_str)
        all_rotations_quat_xyzw = tch.zeros(num_gaussians, 4, device=_device_str)

        for i in range(num_gaussians):
            cov_matrix_g = covariances[i]
            cov_matrix_g_symmetric = (
                cov_matrix_g + cov_matrix_g.transpose(-1, -2)
            ) / 2.0

            eigenvalues, eigenvectors = tch.linalg.eigh(cov_matrix_g_symmetric)
            scales_g = tch.sqrt(
                tch.maximum(eigenvalues, tch.tensor(1e-8, device=_device_str))
            )
            all_scales[i] = scales_g

            rotation_matrix_g = eigenvectors
            rot_matrix_np = rotation_matrix_g.detach().cpu().numpy()
            try:
                quat_xyzw_np = SciPyRotation.from_matrix(rot_matrix_np).as_quat()
                all_rotations_quat_xyzw[i] = tch.from_numpy(quat_xyzw_np).to(
                    _device_str
                )
            except ValueError as e:
                logger.warning(
                    f"SVD/Rotation conversion warning for gaussian {i}: {e}. Using identity quaternion."
                )
                all_rotations_quat_xyzw[i] = tch.tensor(
                    [0.0, 0.0, 0.0, 1.0], device=_device_str
                )

        ply_binary_data = _gaussians_to_ply_bytes_adapted(
            means,
            all_scales,
            all_rotations_quat_xyzw,
            harmonics,
            opacities,
            save_sh_dc_only=True,
        )

    logger.info(
        f"Reconstruction complete. Returning PLY data ({len(ply_binary_data)} bytes)."
    )
    return ply_binary_data
