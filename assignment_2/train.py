from itertools import count
import torch
from torch.utils.data import DataLoader, IterableDataset
from typing import Iterable

from models import RNNModel
from state import TrainingSettings

Criterion = torch.nn.Module  # lame.

VERBOSITY = 10

def make_dataloader(dataset: IterableDataset, batch_size: int):
    return DataLoader(dataset, batch_size=batch_size)

def test(model: RNNModel, dataloader: DataLoader, batch_size: int):
    state = model.state_zero("sequence_length")  # MUTATES!
    total = 0
    correct = 0
    for (i, (input_i, output_i)) in enumerate(dataloader):
        guess_i, state = model(input_i, state)
        total += guess_i.size()[0]
        correct += sum([])  # TODO: figure this out.
        if 0 == i % VERBOSITY:
            print('.', end='')
    print()
    return correct/total

def train(epochs: Iterable[str],
          training_data: IterableDataset,
          test_data: IterableDataset,
          model: RNNModel,
          criterion: Criterion,
          optimizer: torch.optim.Optimizer,
          settings: TrainingSettings):
    model.train()
    training_dataloader = make_dataloader(training_data, settings.batch_size)
    test_dataloader = make_dataloader(test_data, settings.batch_size)

    for epoch_name in epochs:
        state = model.state_zero("sequence_length")  # MUTATES!
        for (i, (input_i, output_i)) in enumerate(training_dataloader):
            optimizer.zero_grad()
            guess_i, state = model(input_i, state)
            loss = criterion(guess_i.transpose(1, 2),  # why?
                             output_i)
            # The tutorial says to detach state here, but I think it's wrong.
            loss.backward()
            optimizer.step()
            if 0 == i % VERBOSITY:
                print(f'  epoch: {epoch_name},  batch: {i},  loss: {loss.item()}')
        accuracy = test(model, test_dataloader, settings.batch_size)
        print(f'Epoch {epoch_name} finished with accuracy {accuracy}.')
        print()
