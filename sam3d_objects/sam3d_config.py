from dataclasses import dataclass
from omegaconf import OmegaConf

@dataclass
class SAM3DConfig:
    config_path: str = ""
    seed: int = 42
    stage1_steps: int = 25
    stage1_only: bool = False
    with_layout_postprocess: bool = False
    with_mesh_postprocess: bool = False
    with_texture_baking: bool = False
    use_vertex_color: bool = True
    gs_backend: str = "gsplat"