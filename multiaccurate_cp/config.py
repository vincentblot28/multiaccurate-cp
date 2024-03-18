from typing import Optional
from pydantic import BaseModel


class PatchesConfig(BaseModel):
    data_path: str
    patch_size: Optional[int] = 572
    pad_size: Optional[int] = 92
    overlap: Optional[int] = 92


class ModelConfig(BaseModel):
    initial_filters_nb: int = 32
    dropout_rate: float = 0.1
    epochs: int = 100
    lr: float = 0.001
    batch_size: int = 40
    num_workers: int = 8
    patience: int = 100


class TrainConfig(BaseModel):
    ml_data_dir: str
    output_dir: str
    mean_RGB_values_path: str
    model: ModelConfig = ModelConfig()


class InferConfig(BaseModel):
    model_dir: str
    model_name: str
    data_dir: str
    ml_set: str
    output_dir: str
    mean_RGB_values_path: str
