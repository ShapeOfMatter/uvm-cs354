from itertools import count
import torch
from torch.utils.data import DataLoader, IterableDataset

from models import RNNModel
from state import Settings


def make_dataloader(dataset: IterableDataset, batch_size: int):
    return DataLoader(dataset, batch_size=batch_size)

def train(training_data: IterableDataset, test_data: IterableDataset, model: RNNModel, settings: Settings):
    model.train()
    training_dataloader = make_dataloader(training_data, settings.batch_size)
    test_dataloader = make_dataloader(test_data, settings.batch_size)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=settings.training.learning_rate,
                                 betas=(settings.training.beta_1, settings.training.beta_2),
                                 eps=settings.training.epsilon,
                                 weight_decay=settings.training.weight_decay)

    for epoch_num in count():
        state = model.state_zero("sequence_length")  # MUTATES!
        for (batch_num, (input_i, output_i)) in enumerate(dataloader):
            optimizer.zero_grad()
            guess_i, state = model(input_i, state)
            loss = criterion(guess_i.transpose(1, 2),  # why?
                             output_i)
            # TODO: detach state
            loss.backward()
            optimizer.step()
            # TODO: print.
