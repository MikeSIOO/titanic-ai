from dataclasses import dataclass, field
from typing import List

# конфиг по созданию нейросети
@dataclass
class MLPConfig:
    hidden_sizes: List[int] = field(default_factory=lambda: [32]) # список с размером внутренних слоев
    activations: List[str] = field(default_factory=lambda: ['relu']) # список с функциями активации для каждого слоя (если передана только 1 активация, применяем ее ко всем слоям)
    dropout: float = 0.0 # значение Dropout, пименяемое к каждому слою (если <= 0 - игнорируется)
    batchnorm: bool = False # BatchNorm1d, применяемая к каждому слою, если True

# конфиг по созданию оптимизатора, шедурера и процесса тренировки
@dataclass
class TrainingConfig:
    lr: float = 1e-2 # lr при создании оптимизатора
    weight_decay: float = 0 # затухание весов в оптиимзаторе
    optimizer: str = 'sgd' # модель оптимизатора (смотреть optimizer_factory для возможных значений)
    scheduler: str = 'none' # модель шедулера (смотреть scheduler_factory для возможных значений)
    num_epochs: int = 20 # количество эпох для обучения
    batch_size: int = 64 # размер батча для создания дата лоудеров

# общий конфиг для обучения модели
@dataclass
class ExperimentConfig:
    mlp: MLPConfig = field(default_factory=MLPConfig) # конфиг нейросети
    training: TrainingConfig = field(default_factory=TrainingConfig) # конфиг обучения