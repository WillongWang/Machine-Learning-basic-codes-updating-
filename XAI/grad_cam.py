#!/usr/bin/env python3

# ============================
# Description: a grad_cam example code.
# Author: Lin Zhi
# Date: 20 Nov 2023
# ============================

import torch
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from PIL import Image
import json

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = "cuda" if torch.cuda.is_available() else "cpu"
as_numpy = lambda x: x.detach().cpu().numpy()

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

jet = matplotlib.colormaps["jet"] # https://matplotlib.org/stable/gallery/color/colormap_reference.html
jet_colors = jet(np.arange(256))[:, :3] # https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.Colormap.html#matplotlib.colors.Colormap

def plot_result(ax, img, saliency, label = ""): # img为RGB, 见下saliency shape (1, 1, H, W)
    img = np.array(img, dtype = float) / 255.0

    saliency = F.interpolate(saliency, size = img.shape[:2], mode = "bilinear") # size为img的(Height,Weight)
    saliency = as_numpy(saliency)[0, 0]
    saliency = saliency - saliency.min()
    saliency = np.uint8(255 * saliency / saliency.max())
    heatmap = jet_colors[saliency] # debug看shape
    ax.imshow(0.5 * heatmap + 0.5 * img)
    ax.axis("off")
    ax.set_title(label)

# np.uint8()：https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.single:
# class numpy.generic[source]
# Base class for numpy scalar types.
# Class from which most (all?) numpy scalar types are derived. For consistency, exposes the same API as ndarray, despite many consequent attributes being either “get-only,” or completely irrelevant. This is the class from which it is strongly suggested users should derive custom scalar types.

# define the preprocessing transform
image_shape = (224, 224)

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            # mean and stddev of ImageNet.  You might consider using
            # mean and stddev depends on your data
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

with open("data/imagenet_class_index.json") as f:
    indx2label = json.load(f)


def decode_predictions(preds, k=5):
    # return the top k results in the predictions
    return [
        [(*indx2label[str(i)], i, pred[i]) for i in pred.argsort()[::-1][:k]] # [::-1] - https://numpy.org/doc/stable/user/basics.indexing.html
        for pred in as_numpy(preds)
    ]

class Probe:
    def get_hook(self,):
        self.data = []
        def hook(module, input, output):
            self.data.append(output)
        return hook #


# load the image
print("loading the image...")
img = Image.open("./data/dog.png")
#img = Image.open("./data/zebra.png")
#img = Image.open("./data/boat.png")
img = img.convert("RGB")

x = transform(img).unsqueeze(0)  # transform and reshape it to [1, C, *image_shape] (image_shapes是height(H),weight(W))
x = x.to(device)

print("loading the model...")
### You can change the model here.
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
# model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

model.eval()
model.to(device)

#add a probe to model
probe = Probe()
# probe will save the output of the layer4 during forward
# The hook will be called every time a gradient with respect to the Tensor is computed
# This function returns a handle with a method handle.remove() that removes the hook 
# from the module
handle = model.layer4.register_forward_hook(probe.get_hook()) # resnet.py

logits = model(x) # resnet.py, logits形状为(1,num_classes)
preds = logits.softmax(-1)

print("the prediction result:")
for tag, label, i, prob in decode_predictions(preds)[0]:
    print(f"{tag} {label:16} {i:5} {prob:6.2%}")


N = 3 # top-N to draw saliency map

targets = preds.argsort(descending=True)[0, :N]
for i, target in enumerate(targets):
    print(f"Calculating the saliency of the {i+1}-th likely class...")
    target = target.item()
    ### Grad_Cam
    # get the last_conv_output (上layer4)
    last_conv_output = probe.data[0]
    last_conv_output.retain_grad() #make sure the intermediate result save its grad

    #backprop
    logits[0, target].backward(retain_graph=True)
    grad = last_conv_output.grad # shape (1, C, H, W)
    #taking average on the H-W panel
    weight = grad.mean(dim = (-1, -2), keepdim = True) # (1, C, 1, 1)
    saliency = (last_conv_output * weight).sum(dim = 1, keepdim = True) # channel维吧, saliency shape (1, 1, H, W)
    #relu
    saliency = saliency.clamp(min = 0)
    last_conv_output.grad.zero_()

    ax = plt.subplot(N, 1, i + 1)
    plot_result(ax, img, saliency, "grad_cam on {} {}".format(*indx2label[str(target)]))
handle.remove()
plt.savefig('output_image.png')