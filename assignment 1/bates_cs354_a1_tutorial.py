from PIL import Image
from pprint import pprint
import torch
from torchvision import transforms
from typing import Iterable



DOGFILE = 'dog.jpg'
preprocess = transforms.Compose([
    #transforms.Resize(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
HAS_CUDA = torch.cuda.is_available()
with open("imagenet_classes.txt", "r") as f:
    IMAGENET_CLASSES = tuple(s.strip() for s in f.readlines())

def fetch_model():
    model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
    model.eval()
    # move the input and model to GPU for speed if available
    if HAS_CUDA:
        model.to('cuda')
    return model

def prep_image(filename: str):
    i = Image.open(filename)
    t = preprocess(i)
    return t

def make_batches(filenames: Iterable[str]):
    input_tensors = map(prep_image, filenames)
    mini_batches = (t.unsqueeze(0) for t in input_tensors) # I think this is makeing batches of size zero.
    if HAS_CUDA:
        yield from (b.to('cuda') for b in mini_batches)
    else:
        yield from mini_batches

def get_output_vector(model, batch):
    with torch.no_grad():
        return model(batch)

def get_top_n_probabilities(output_vector, n):
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output_vector[0], dim=0)
    probs, catids = torch.topk(probabilities, n)
    return {cid: prob.item() for (cid, prob) in zip(catids, probs)}

def main():
    model = fetch_model()
    batch, *_ = make_batches([DOGFILE])
    output_vector = get_output_vector(model, batch)
    top5 = get_top_n_probabilities(output_vector, 5)
    pprint({IMAGENET_CLASSES[cid]: probability for (cid, probability) in top5.items()})

if __name__ == "__main__":
    main()
