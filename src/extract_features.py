"""
The comment below are just explanation for each function. For me to easy to understand.
Those comment are human-written by ME!!! please delete it before submission or I will hunt you down!
"""

import torch
from PIL import Image
from torchvision import transforms

def extract_features(model, image_input):
    model.eval() # turn the model in evaluation mode

    features = [] # Array that drop the output from the Conv layer

    """
    Hook function allow us to attach custom functions (hooks) to tensors and modules within your neural network, 
    enabling you to monitor, modify, or record various aspects of the computation graph.
    """
    def hook_fn(module, input, output):
        features.append(output) #grap the output and append the feature array

    for layer in model.modules(): #<----- Bro this is the hook. This cooks our dinner.
        if isinstance(layer, torch.nn.Conv2d): # See the layer is it Conv layer
            # we are doing forward hooks by calling "hook_fn" this already included model input and output
            layer.register_forward_hook(hook_fn)

    """
    Forward hooks are executed after the forward pass through a layer is completed but before the output is returned. 
    They provide access to both the input and the output of the layer. 
    This allows you to inspect or modify the data flowing through the layer during the forward pass.

    Forward hooks can be used for:
    Visualize activations or feature maps.
    Compute statistics on the activations.
    Perform any custom operation on the layer's output.
    
    Well we do have 3 kinds of hooks where it is forward Pre-Hooks, Backward hooks and the one we are using.
    """

    # --- NEW LOGIC: Check if it's a file path (string) or a dummy image (tensor) ---
    if isinstance(image_input, str):
        # It's a file path, load the real image
        image = Image.open(image_input).convert('L')
        transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor()
        ])
        input_tensor = transform(image).unsqueeze(0) # Shape: [1, 1, 96, 96]
    else:
        # It's already a tensor (dummy imagine), use it directly
        # -> 1 is batch size 1 is channel and 96 x 96 is image size
        input_tensor = image_input

    model(input_tensor) # activatae the model using input_tensor (dummy or real)

    return features