from typing import Optional
from pydantic import BaseModel


class PatchesConfig(BaseModel):
    data_path: str
    patch_size: Optional[int] = 572
    pad_size: Optional[int] = 92
    overlap: Optional[int] = 92


class UnetConfig(BaseModel):
    initial_filters_nb: int = 32
    dropout_rate: float = 0.1
    epochs: int = 100
    lr: float = 0.001
    batch_size: int = 40
    num_workers: int = 8
    patience: int = 20


class TrainUnetConfig(BaseModel):
    ml_data_dir: str
    output_dir: str
    mean_RGB_values_path: str
    model: UnetConfig = UnetConfig()


class InferUnetConfig(BaseModel):
    model_dir: str
    model_name: str
    data_dir: str
    ml_set: str
    output_dir: str
    mean_RGB_values_path: str


class ResnetConfig(BaseModel):
    resnet: str = "resnet18"
    embedding_size: int = 256
    target_recall: float = 0.9
    model_input: str = "images"
    lr: float = 0.0001
    batch_size: int = 40
    num_workers: int = 8
    patience: int = 20
    epochs: int = 100


class TrainResConfig(BaseModel):
    ml_data_dir: str
    probas_dir: str
    output_dir: str
    mean_RGB_values_path: str
    model: ResnetConfig = ResnetConfig()


class InferResConfig(BaseModel):
    model_dir: str
    model_name: str
    data_dir: str
    pred_proba_dir: str
    ml_set: str


class InferPolypConfig(BaseModel):
    model_dir: str
    data_dir: str
    ml_set: str
    output_dir: str
