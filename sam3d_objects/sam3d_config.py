from dataclasses import dataclass
from omegaconf import OmegaConf

@dataclass
class SAM3DConfig:
    config_path: str = ""
    seed: int = 42
    ss_step: int = 25
    ss_only: bool = False