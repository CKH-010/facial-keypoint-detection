"""
The comment below are just explanation for each function. For me to easy to understand.
Those comment are human-written by ME!!! please delete it before submission or I will hunt you down!
"""

import torch
from PIL import Image
from torchvision import transforms

def extract_features(model, image_input):
    model.eval()
    features = []

    if isinstance(image_input, str):
        image = Image.open(image_input).convert("L")
        transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
        ])
        input_tensor = transform(image).unsqueeze(0)
    else:
        input_tensor = image_input

    input_tensor = input_tensor.to(next(model.parameters()).device)

    def hook_fn(module, inputs, output):
        features.append(output.detach())

    handles = [
        layer.register_forward_hook(hook_fn)
        for layer in model.modules()
        if isinstance(layer, torch.nn.Conv2d)
    ]

    try:
        with torch.no_grad():
            model(input_tensor)
    finally:
        for handle in handles:
            handle.remove()

    return features