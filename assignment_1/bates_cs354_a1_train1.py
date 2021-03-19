from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from PIL import Image
import sys
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from typing import Iterable, Tuple

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
HAS_CUDA = torch.cuda.is_available() and 1 < torch.cuda.device_count()
LOCATION = torch.device('cuda') if HAS_CUDA else torch.device('cpu')

def fetch_model(settings: Settings, worker: Worker):
    m_pretrained = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
    m_untrained = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=False)
    model = m_pretrained if settings.pretrained_features else m_untrained
    model.classifier = m_pretrained.classifier if settings.pretrained_classifier else m_untrained.classifier
    classes = len(listdir(settings.training_source_dir))
    worker.log(f'Found {classes} classes.')
    model.classifier[6] = torch.nn.Linear(4096, classes)
    for (layer, trainable) in chain(zip(model.features, settings.trainable_features),
                                    zip(model.classifier, settings.trainable_classifier)):
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            for param in layer.parameters():
                param.requires_grad = trainable
    model.eval()
    model.to(LOCATION)
    parallel_model = torch.nn.DataParallel(model) if HAS_CUDA else model
    return parallel_model

def make_trainer(model, settings: TrainingSettings):
    return (optim.SGD(model.parameters(),
                      lr=settings.learning_rate,
                      momentum=settings.momentum,
                      weight_decay=settings.weight_decay),
            torch.nn.CrossEntropyLoss())

def train(model, batches, optimizer, criterion, epoch, worker: Worker, device=LOCATION):
    model.train()
    for (batch_num, (data, target)) in enumerate((d.to(device), t.to(device)) for (d, t) in batches):
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()
        log_format = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
        if batch_num % 25 == 0:
            worker.log(log_format.format(epoch,
                                         batch_num * len(data),
                                         len(batches.dataset),
                                         100. * batch_num / len(batches),
                                         loss.item()))

def test(model, batches, criterion, worker: Worker, device=LOCATION):
    model.eval()
    with torch.no_grad():
        def _test(data, target):
            result = model(data)
            predicates = result.argmax(dim=1, keepdim=True)
            return (criterion(result, target).item(),
                    predicates.eq(target.view_as(predicates)).sum().item())
        losses_successes = [_test(data.to(device), target.to(device))
                            for (data, target) in batches]
        loss = sum(l for (l, s) in losses_successes) / len(losses_successes)
        successes = sum(s for (l, s) in losses_successes)
        log_format = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
        worker.log(log_format.format(loss,
                                     successes,
                                     len(batches.dataset),
                                     100. * successes / len(batches.dataset)))
        return successes / len(batches.dataset)

def make_batches(data_directory: str, max_batch_size: int, worker: Worker):
    images = ImageFolder(root=data_directory, transform=preprocess)
    batch_size = next(n
                      for n
                      in range(max_batch_size, 0, -1)
                      if (len(images) % n) == 0)
    worker.log(f'Using batch-size {batch_size}')
    return DataLoader(i,
                      batch_size=batch_size,
                      shuffle=True,
                      pin_memory=HAS_CUDA,)

def epoch(name, model, training_data, testing_data, worker: Worker):
    optimizer, criterion = make_trainer(model, worker.trainer)
    train(model, testing_data, optimizer, criterion, name, worker)
    accuracy = test(model, testing_data, criterion, worker)
    return accuracy

def main(settings_file):
    settings = get_settings(settings_file)
    with initialize_and_recure(settings) as worker:
        worker.log(f"Hi I'm {worker.name}! CUDA: {HAS_CUDA}. Location: {LOCATION}. gpus: {torch.cuda.device_count()}")
        model = fetch_model(settings, worker)
        train_batch = make_batches(settings.training_source_dir, settings.max_batch_size, worker)
        test_batch = make_batches(settings.testing_source_dir, settings.max_batch_size, worker)
        for e in worker.epochs(settings):
            worker.upkeep_model(model, settings)
            accuracy = epoch(e, model, train_batch, test_batch, worker)
            upkeep_state(worker, model, accuracy)

if __name__ == "__main__":
    main(sys.argv[1])
