from itertools import count
import sys
from time import time
import torch
from typing import Iterable

from models import RNNModel
from state import Settings, TrainingSettings
from train import Criterion

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
    model = RNNModel(embedding_width=settings.model_settings.embedding_width,
                     hidden_width=settings.model_settings.hidden_width,
                     rnn_height=settings.model_settings.rnn_height)
    criterion = make_criterion()
    optimizer = make_optimizer(model, settings.training_settings)
    
    #    train(epochs(settings), 

if __name__ == "__main__":
    main(sys.argv[1])
