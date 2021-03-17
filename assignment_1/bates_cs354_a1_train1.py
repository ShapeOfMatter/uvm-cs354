import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pprint import pprint
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from typing import Iterable, Tuple



DOGFILE = 'dog.jpg'
SKETCHDIR = 'sketch_subset'
MODEL_FILE = 'model.pt'
preprocess = transforms.Compose([
    #transforms.Resize(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
HAS_CUDA = torch.cuda.is_available() and 1 < torch.cuda.device_count()
LOCATION = torch.device('cuda') if HAS_CUDA else torch.device('cpu')

def show_image(image, title):
    img = image.numpy().transpose((1,2,0))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    de_normalized = np.clip(img*std + mean, 0, 1)
    plt.imshow(de_normalized)
    plt.title(title)
    plt.pause(0.001) # supposedly needed.

def log(thing):
    if isinstance(thing, str):
        print(thing, flush=True)
    else:
        pprint(thing)

def fetch_model(classes: int, *, pretrained=True, resume=False):
    model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=pretrained)
    model.classifier[6] = torch.nn.Linear(4096, classes)
    if resume:
        model.load_state_dict(torch.load(MODEL_FILE), strict=True)
    for param in model.features.parameters():
        param.requires_grad = False
    model.eval()
    model.to(LOCATION)
    return torch.nn.DataParallel(model) if HAS_CUDA else model

def make_trainer(model, learning_rate=0.01, momentum=0.5, weight_decay=0.0005):
    return (optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay),
            torch.nn.CrossEntropyLoss())

def train(model, batches, optimizer, criterion, epoch, device=LOCATION):
    model.train()
    for (batch_num, (data, target)) in enumerate((d.to(device), t.to(device)) for (d, t) in batches):
        optimizer.zero_grad()
        result = model(data)
        loss = criterion(result, target)
        loss.backward()
        optimizer.step()
        if batch_num % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_num * len(data),  # wrong, batches are not the same size!
                                                                           len(batches.dataset),
                                                                           100. * batch_num / len(batches),
                                                                           loss.item()))

def test(model, batches, criterion, device=LOCATION):
    model.eval()
    with torch.no_grad():
        #(batch_num, (data, target)) = next(enumerate((d.to(device), t.to(device)) for (d, t) in batches))
        #image_grid = make_grid(data[:4], nrow=4)
        #show_image(image_grid, target[:4])
        #pprint(model(data))
        #pprint(target)
        #pprint(criterion(model(data), target))
        #predicates = model(data).argmax(dim=1, keepdim=True)
        #pprint(target.view_as(predicates))
        #pprint(predicates.eq(target.view_as(predicates)))
        #input("anykey")
        def _test(data, target):
            result = model(data)
            predicates = result.argmax(dim=1, keepdim=True)
            return (criterion(result, target).item(),
                    predicates.eq(target.view_as(predicates)).sum().item())
        losses_successes = [_test(data.to(device), target.to(device))
                            for (data, target) in batches]
        loss = sum(l for (l, s) in losses_successes) / len(losses_successes)
        successes = sum(s for (l, s) in losses_successes)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(loss,
                                                                                     successes,
                                                                                     len(batches.dataset),
                                                                                     100. * successes / len(batches.dataset)))
        return loss

def make_batches(data_directory: str, ratios=(1.0, ), batch_size=64):
    images = ImageFolder(root=data_directory,
                         transform=preprocess,
                         #target_transform: Union[Callable, NoneType] = None,
                         #loader: Callable[[str], Any] = <function default_loader>,
                         #is_valid_file: Union[Callable[[str], bool], NoneType] = None
                         )
    sizes = [int(r * len(images) / sum(ratios)) for r in ratios]
    assert len(images) == sum(sizes)
    return [DataLoader(i,
                      batch_size=batch_size,
                      shuffle=False,  # default, probably fine.
                      sampler=None,  # default, necessary for iter dataset.
                      batch_sampler=None,  # default, necessary for iter dataset.
                      num_workers=0,  # default, possibly not optimal, but reliable.
                      collate_fn=None,  # default, I don't think i want to write my own.
                      pin_memory=HAS_CUDA,
                      drop_last=False,  # default, fine
                      timeout=0,  # default
                      worker_init_fn=None,  # default
                      prefetch_factor=2,  # default
                      persistent_workers=False  # default
            ) for i in random_split(images, sizes)]

def get_output_vector(model, batch):
    with torch.no_grad():
        return model(batch)

def get_top_n_probabilities(output_vector, n):
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = F.softmax(output_vector[0], dim=0)
    probs, catids = torch.topk(probabilities, n)
    return {cid: prob.item() for (cid, prob) in zip(catids, probs)}

def main():
    print(f'Hello world! CUDA? {HAS_CUDA} Location? {LOCATION}')
    print(f'gpus: {torch.cuda.device_count()}')
    model = fetch_model(250)
    train_batch, test_batch = make_batches(SKETCHDIR, (90, 10), batch_size=60)
    learning_rate = 0.01
    optimizer, criterion = make_trainer(model, learning_rate=learning_rate, momentum=0.5)
    losses=[2]
    for epoch in range(100):
        train(model, train_batch, optimizer, criterion, epoch)
        loss = test(model, test_batch, criterion)
        if 0.1 > abs(loss - np.average(losses[-4:])):
            learning_rate *= 0.1
            optimizer, criterion = make_trainer(model, learning_rate=learning_rate, momentum=0.5)
        print(learning_rate)
        losses.append(loss)
        losses = losses[-10:]
        pprint(losses)
    torch.save(model.state_dict(), MODEL_FILE)
    return model

if __name__ == "__main__":
    main()
