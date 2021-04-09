from itertools import count
import sys
from time import time
import torch
from typing import Iterable

from datasets import SerialCharData
from models import RNNModel
from state import Settings, TrainingSettings
from train import Criterion, train

def make_criterion() -> Criterion:
    return torch.nn.CrossEntropyLoss()

def make_optimizer(model: RNNModel, settings: TrainingSettings) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(),
                            lr=settings.learning_rate,
                            betas=(settings.beta_1, settings.beta_2),
                            eps=settings.epsilon,
                            weight_decay=settings.weight_decay)

def epochs(settings: Settings) -> Iterable[str]:
    end = time() + settings.total_lifetime
    es = count()
    while time() < end:
        yield f'e_{next(es)}_{settings.name}'

def main(settings_filename: str):
    settings = Settings.load(settings_filename)
    training_settings = settings.training_settings
    model = RNNModel(f'm_{settings.name}', settings.model_settings)
    criterion = make_criterion()
    optimizer = make_optimizer(model, training_settings)
    training_data = SerialCharData(settings.training_file, training_settings.snippet_length).as_dataloader(training_settings.batch_size)
    test_data = SerialCharData(settings.test_file, training_settings.snippet_length).as_dataloader(training_settings.batch_size)
    
    train(epochs=epochs(settings),
          training_data=training_data,
          test_data=test_data,
          model=model,
          criterion=criterion,
          optimizer=optimizer,
          settings=training_settings)

if __name__ == "__main__":
    main(sys.argv[1])
