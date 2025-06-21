# server/vggt_service.py

import os
import subprocess
import shutil
import uuid
from pathlib import Path
from typing import List
from PIL import Image
import pycolmap

def convert_colmap_to_ply(colmap_sparse_path: Path, ply_output_path: Path):
    """
    Converts a COLMAP sparse reconstruction to a .ply file.
    """
    reconstruction = pycolmap.Reconstruction(colmap_sparse_path)
    reconstruction.export_ply(ply_output_path)

class VGGTService:
    def __init__(self, model_path="server/VGGT"):
        self.model_path = Path(model_path).resolve()
        self.output_root = Path("output_jobs").resolve()
        self.output_root.mkdir(exist_ok=True)

    def reconstruct_scene(self, images: List[Image.Image], scene_name: str = "reconstruction") -> bytes:
        """
        Generates a .ply file from a list of images by running the VGGT process.

        Args:
            images (List[Image.Image]): A list of PIL images.
            scene_name (str): A name for the scene.

        Returns:
            bytes: The binary content of the generated .ply file.
        """
        if not images:
            raise ValueError("At least one image is required for reconstruction.")

        job_id = str(uuid.uuid4())
        job_path = self.output_root / job_id
        
        # 1. Prepare the data directory structure required by VGGT
        vggt_image_path = job_path / "images"
        vggt_image_path.mkdir(parents=True, exist_ok=True)

        # Save uploaded images to the job directory
        for i, img in enumerate(images):
            img.save(vggt_image_path / f"{i:04d}.png")

        # 2. Construct the training command
        working_dir = self.model_path.parent.parent # tools/view-to-3dgs
        script_path = self.model_path / "demo_colmap.py"
        
        cmd = [
            "python", str(script_path),
            "--scene_dir", str(job_path),
        ]

        # 3. Execute the command
        try:
            print(f"Running VGGT for job {job_id}: uv run --active {' '.join(cmd)}")
            process = subprocess.Popen(["uv", "run", "--active"] + cmd, cwd=working_dir, 
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

            # Stream the output
            for line in iter(process.stdout.readline, ''):
                print(line, end='')
            
            process.stdout.close()
            return_code = process.wait()

            if return_code:
                raise subprocess.CalledProcessError(return_code, cmd)

        except subprocess.CalledProcessError as e:
            print(f"Error during VGGT execution for job {job_id}: {e}")
            raise RuntimeError("Failed to execute VGGT reconstruction process.") from e

        # 4. Locate and convert the output file
        colmap_sparse_path = job_path / "sparse"
        points_file = colmap_sparse_path / "points3D.bin"

        if not points_file.exists():
            raise FileNotFoundError(f"Could not find the COLMAP points file at {points_file}")

        ply_output_path = job_path / f"{scene_name}.ply"
        convert_colmap_to_ply(colmap_sparse_path, ply_output_path)
        
        if not ply_output_path.exists():
            raise FileNotFoundError(f"Could not find the converted .ply file at {ply_output_path}")

        output_data = ply_output_path.read_bytes()

        # 5. Clean up the job directory
        shutil.rmtree(job_path)

        return output_data
