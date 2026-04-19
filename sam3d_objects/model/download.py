import os
import shutil
from huggingface_hub import snapshot_download

def setup_checkpoints(path: str) -> str:
    """
    Downloads the sam-3d-objects model and organizes it into the target path.
    
    Args:
        path: The final directory where the weights should live 
                     (e.g., 'checkpoints/hf' or 'checkpoints/sam3d')
    """
    # Create the temporary download path dynamically based on your target
    # e.g., "checkpoints/sam3d" -> "checkpoints/sam3d-download"
    temp_dir = f"{path}-download"
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    submodule_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    patch_dir = os.path.join(submodule_root, "checkpoints", "hf")

    # Flag to check if we actually need to download
    needs_download = not (os.path.exists(path) and os.listdir(path))
    
    if needs_download:
        print(f"[Fast-SAM3D] Downloading weights to '{temp_dir}'...")
        try:
            snapshot_download(
                repo_id="facebook/sam-3d-objects",
                repo_type="model",
                local_dir=temp_dir,
                max_workers=1,
                allow_patterns="checkpoints/*"  
            )

            source_checkpoints_dir = os.path.join(temp_dir, "checkpoints")

            print(f"[Fast-SAM3D] Moving weights to '{path}'...")
            if os.path.exists(path):
                shutil.rmtree(path) 
                
            shutil.move(source_checkpoints_dir, path)

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    else:
        print(f"[Fast-SAM3D] Checkpoints already exist at '{path}'.")

    # 2. ALWAYS apply the patches (Configs might have been updated in Git)
    if os.path.exists(patch_dir):
        print(f"[Fast-SAM3D] Applying config patches from '{patch_dir}' to '{path}'...")
        # dirs_exist_ok=True merges the folders and overwrites identically named files
        shutil.copytree(patch_dir, path, dirs_exist_ok=True)
    else:
        print(f"[Fast-SAM3D] Warning: No patch directory found at '{patch_dir}'!")

    print(f"[Fast-SAM3D] Successfully set up checkpoints at '{path}'")
    return path