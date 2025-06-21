# server/vggt_service.py

import os
import subprocess
import shutil
import uuid
from pathlib import Path
from typing import List
from PIL import Image

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

        # 4. Locate and read the output file
        print("VGGT process finished. Reading .ply file...")
        ply_path = job_path / "sparse" / "points.ply"

        print(f"Checking for .ply file at: {ply_path}")
        if not ply_path.exists():
            raise FileNotFoundError(f"Could not find the .ply file at {ply_path}")

        print("Reading .ply file content...")
        output_data = ply_path.read_bytes()

        # 5. Clean up the job directory
        print(f"Cleaning up job directory: {job_path}")
        shutil.rmtree(job_path)
        print("Cleanup complete.")

        return output_data
