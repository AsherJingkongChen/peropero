# server/instantsplat_service.py

import os
import subprocess
import shutil
import uuid
from pathlib import Path
from typing import List
from PIL import Image

class InstantSplatService:
    def __init__(self, model_path="server/InstantSplat"):
        self.model_path = Path(model_path).resolve()
        self.output_root = Path("output_jobs").resolve()
        self.output_root.mkdir(exist_ok=True)

    def reconstruct_scene(self, images: List[Image.Image], scene_name: str = "reconstruction") -> bytes:
        """
        Generates a .ply file from a list of images by running the InstantSplat training process.

        Args:
            images (List[Image.Image]): A list of PIL images.
            scene_name (str): A name for the scene.

        Returns:
            bytes: The binary content of the generated .ply file.
        """
        if len(images) < 3:
            raise ValueError(f"InstantSplat requires at least 3 images for reconstruction, but only {len(images)} were provided.")

        job_id = str(uuid.uuid4())
        job_path = self.output_root / job_id
        
        # 1. Prepare the data directory structure required by InstantSplat
        instantsplat_image_path = job_path / "images"
        instantsplat_image_path.mkdir(parents=True, exist_ok=True)

        # Save uploaded images to the job directory
        for i, img in enumerate(images):
            img.save(instantsplat_image_path / f"{i:04d}.png")

        # 2. Construct the training commands to be executed on the remote server
        train_iterations = 500
        model_output_path = job_path / "output"
        model_output_path.mkdir(parents=True, exist_ok=True)
        
        # Note: InstantSplat scripts expect to be run from within the InstantSplat directory
        working_dir = self.model_path

        # Command for geometry initialization
        init_cmd = [
            "uv", "run", "python", "./init_geo.py",
            "-s", str(job_path),
            "-m", str(model_output_path),
            "--n_views", str(len(images)),
            "--focal_avg",
            # "--co_vis_dsp",
            "--conf_aware_ranking",
        ]

        # Command for training
        train_cmd = [
            "uv", "run", "python", "./train.py",
            "-s", str(job_path),
            "-m", str(model_output_path),
            "-r", "1",
            "--n_views", str(len(images)),
            "--iterations", str(train_iterations),
            # "--pp_optimizer",
            # "--optim_pose",
            "--save_iterations", str(train_iterations),
        ]

        # 3. Execute the commands sequentially in a blocking manner
        try:
            print(f"Running initialization for job {job_id}: {' '.join(init_cmd)}")
            subprocess.run(init_cmd, check=True, cwd=working_dir)

            print(f"Running training for job {job_id}: {' '.join(train_cmd)}")
            subprocess.run(train_cmd, check=True, cwd=working_dir)
        except subprocess.CalledProcessError as e:
            print(f"Error during InstantSplat execution for job {job_id}: {e}")
            raise RuntimeError("Failed to execute InstantSplat reconstruction process.") from e

        # 4. Locate and read the output .ply file
        final_ply_path = model_output_path / "point_cloud" / f"iteration_{train_iterations}" / "point_cloud.ply"
        
        if not final_ply_path.exists():
            raise FileNotFoundError(f"Could not find the final .ply file at {final_ply_path}")

        ply_binary_data = final_ply_path.read_bytes()

        # 5. Clean up the job directory
        shutil.rmtree(job_path)

        return ply_binary_data
