import os
import torch
import numpy as np
from pytorch3d.transforms import (
    Rotate,
    Translate,
    Scale,
    Transform3d,
    quaternion_to_matrix,
)
from sam3d_objects.pipeline.inference_pipeline_pointmap import camera_to_pytorch3d_camera


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


def canonical_to_world(output, c2w):
    """
    Transform mesh vertices from canonical space to world space.

    Args:
        output: Predictor output dict with 'scale', 'translation', 'rotation', 'mesh'
        c2w: 4x4 camera-to-world matrix

    Returns:
        verts_w: Transformed vertices in world space
    """
    object_scale = output['scale']
    translation = output['translation']
    rotation = output['rotation']

    device = object_scale.device
    Rotation = quaternion_to_matrix(rotation.squeeze(1))
    tfm_ori = compose_transform(scale=object_scale, rotation=Rotation, translation=translation)

    verts_w_in_canonical = torch.tensor(output['glb'].vertices, dtype=torch.float32, device=device) @ torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float32, device=device).T

    verts_w = verts_w_in_canonical.clone()
    verts_w = tfm_ori.transform_points(verts_w)

    camera_convention_transform = (
        Transform3d()
        .rotate(camera_to_pytorch3d_camera(device=device).rotation)
        .to(device=device)
    )
    verts_w_c = camera_convention_transform.inverse().transform_points(verts_w)

    R_wc = torch.tensor(c2w[:3, :3], dtype=torch.float32).to(device)
    T_wc = torch.tensor(c2w[:3, 3], dtype=torch.float32).to(device)

    verts_w = (R_wc @ verts_w_c.T).T + T_wc

    return verts_w


def export_world_mesh(output, c2w, output_path):
    """
    Transform mesh vertices from canonical space to world space and export to GLB.

    Args:
        output: Predictor output dict containing 'scale', 'translation', 'rotation', 'mesh', 'glb'
        c2w: 4x4 camera-to-world matrix
        output_path: Path to save the GLB file
    """
    verts_w = canonical_to_world(output, c2w)
    output['glb'].vertices = verts_w.cpu().numpy()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output["glb"].export(output_path)