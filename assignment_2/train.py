from itertools import count
from time import time
import torch
from torch.utils.data import DataLoader
from typing import Iterable

from models import RNNModel
from state import TrainingSettings

Criterion = torch.nn.Module  # lame.

VERBOSITY = 1000

def test(model: RNNModel, dataloader: DataLoader, settings: TrainingSettings):
    model.eval()
    state = model.state_zero(settings.snippet_length)  # MUTATES!
    total = 0
    correct = 0
    for (i, (input_i, output_i)) in enumerate(dataloader):
        guess_i, state = model(input_i, state)
        total += guess_i.shape[0]
        correct += sum((x == y).item()
                       for (x, y)
                       in zip(output_i.select(1, -1), guess_i.argmax(2).select(1, -1)))
        if 0 == i % VERBOSITY:
            print('.', end='')
    print()
    return correct/total

def train(epochs: Iterable[str],
          training_data: DataLoader,
          test_data: DataLoader,
          model: RNNModel,
          criterion: Criterion,
          optimizer: torch.optim.Optimizer,
          settings: TrainingSettings):
    model.train()
    print(f'Training model {model.name}')
    for epoch_name in epochs:
        state = model.state_zero(settings.snippet_length)  # MUTATES!
        for (i, (input_i, output_i)) in enumerate(training_data):
            optimizer.zero_grad()
            guess_i, state = model(input_i, state)
            loss = criterion(guess_i.transpose(1, 2),  # why?
                             output_i)
            state = model.detach_state(state)  # why?
            loss.backward()
            optimizer.step()
            if 0 == i % VERBOSITY:
                print(f'\tepoch: {epoch_name},\tbatch: {i},\tloss: {loss.item():.6}\t\t{int(time())}')
        print(f'Model {model.name} has finished training epoch {epoch_name}.')
        accuracy = test(model, test_data, settings)
        print(f'Epoch {epoch_name} finished with accuracy {100 * accuracy:.8}%.')
        print()
