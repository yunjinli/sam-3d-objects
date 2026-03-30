import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'notebook'))
from inference import Inference, load_image, load_single_mask
from pytorch3d.transforms import (
    Rotate,
    Translate,
    Scale,
    Transform3d,
    quaternion_to_matrix,
    axis_angle_to_quaternion,
)
import torch
import open3d
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--image-path", type=str, help="Path to the input image")
parser.add_argument("--from-ssh", action="store_true", help="Whether to stream visualization over SSH")
args = parser.parse_args()

def compose_transform(
    scale: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor
) -> Transform3d:
    """
    Args:
        scale: (..., 3) tensor of scale factors
        rotation: (..., 3, 3) tensor of rotation matrices
        translation: (..., 3) tensor of translation vectors
    """
    tfm = Transform3d(dtype=scale.dtype, device=scale.device)
    return tfm.scale(scale).rotate(rotation).translate(translation)

PATH = os.getcwd()
TAG = "hf"
config_path = f"{PATH}/checkpoints/{TAG}/pipeline.yaml"

initialization_start_time = time.time()
inference = Inference(config_path, compile=False)
initialization_end_time = time.time()
# IMAGE_PATH = f"{PATH}/notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png"
# IMAGE_PATH = f"{PATH}/notebook/images/office/image.png"
IMAGE_PATH = os.path.abspath(args.image_path)
IMAGE_NAME = os.path.basename(os.path.dirname(IMAGE_PATH))

image = load_image(IMAGE_PATH)

import glob

# 1. Define your directory
mask_dir = os.path.dirname(IMAGE_PATH)

# 2. Use glob to match the pattern: any numbers followed by .png
# The [0-9]* ensures we only grab files like 0.png, 12.png, etc.
filenames = sorted(glob.glob(os.path.join(mask_dir, '[0-9]*.png')))
# print(mask_files)
outputs = []
forward_times = []
for filename in filenames:
    index = int(os.path.basename(filename.split('.')[0]))
    forward_start_time = time.time()
    # print(f"Loading mask index: {index}")
    
    mask = load_single_mask(os.path.dirname(IMAGE_PATH), index=index)
    output = inference(image, mask, seed=42)
    outputs.append(output)
    forward_end_time = time.time()
    forward_times.append(forward_end_time - forward_start_time)

pcd_scene = open3d.geometry.PointCloud()
pcd_scene.points = open3d.utility.Vector3dVector(output["pointmap"].reshape(-1, 3).cpu().numpy())
pcd_scene.colors = open3d.utility.Vector3dVector(output["pointmap_colors"].reshape(-1, 3).cpu().numpy())

scene_pcd_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.6, origin=[0, 0, 0])

vis_pcd = [pcd_scene, scene_pcd_frame]

compose_scene_start_time = time.time()

for output in outputs:
    object_voxels_canonical = output['voxel']
    object_scale = output['scale']
    translation = output['translation']
    rotation = output['rotation']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Rotation = quaternion_to_matrix(rotation.squeeze(1))
    center = translation[0].clone()
    tfm_ori = compose_transform(scale=object_scale, rotation=Rotation, translation=translation)
    # print(tfm_ori.get_matrix())

    # rotate mesh (from z-up to y-up)
    # object_pcd_metric = object_voxels_canonical.cpu().numpy() @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).T
    object_pcd_metric = object_voxels_canonical.cpu().numpy()
    object_pcd_metric = torch.from_numpy(object_pcd_metric).float().to(device)
    object_pcd_metric = tfm_ori.transform_points(object_pcd_metric.unsqueeze(0))
    object_pcd_metric = object_pcd_metric[0].cpu().numpy()  # pytorch3d, y-up, x left, z inwards.
    pcd_object = open3d.geometry.PointCloud()
    pcd_object.points = open3d.utility.Vector3dVector(object_pcd_metric)
    vis_pcd.append(pcd_object)

compose_scene_end_time = time.time()

print("Initialization time: {:.2f} seconds".format(initialization_end_time - initialization_start_time))
print("Forward pass time: {:.2f} seconds".format(sum(forward_times)))
for i, t in enumerate(forward_times):
    print("Forward pass time for object {}: {:.2f} seconds".format(i, t))
    
print("Compose scene time: {:.2f} seconds".format(compose_scene_end_time - compose_scene_start_time))
if args.from_ssh:
    ## For ssh remote
    os.environ["WEBRTC_IP"] = "127.0.0.1"
    os.environ["WEBRTC_PORT"] = "8888" 
    open3d.visualization.webrtc_server.enable_webrtc()

    print("Starting WebRTC stream...")

    # 2. THE FIX: Use the standard draw function
    open3d.visualization.draw(vis_pcd)
else:
    open3d.visualization.draw_geometries(vis_pcd)

