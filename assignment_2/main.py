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

def periods(settings: Settings) -> Iterable[str]:
    end = time() + settings.total_lifetime
    es = count()
    while time() < end:
        yield f'p_{next(es)}_{settings.name}'

def epochs(period: str, settings: Settings) -> Iterable[str]:
    end = time() + settings.epoch_time
    es = count()
    while time() < end:
        yield f'e_{next(es)}_{period}_{settings.name}'

def main(settings_filename: str):
    settings = Settings.load(settings_filename)
    t_settings = settings.training_settings
    model = RNNModel(f'm_{settings.name}', settings.model_settings)
    criterion = make_criterion()
    optimizer = make_optimizer(model, t_settings)
    training_dataset = SerialCharData(settings.training_file, t_settings.snippet_length)
    test_dataset = SerialCharData(settings.test_file, t_settings.snippet_length)
    
    for period_name in periods(settings):
        train(epochs=epochs(period_name, settings),
              training_data=training_dataset.get_dataloader(batch_size=t_settings.batch_size,
                                                            book_size=t_settings.chunk_size),
              test_data=test_dataset.get_dataloader(batch_size=t_settings.batch_size,
                                                    book_size=t_settings.chunk_size//100),
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              settings=t_settings)
        torch.save(model.state_dict(), settings.model_filename)

if __name__ == "__main__":
    main(sys.argv[1])
