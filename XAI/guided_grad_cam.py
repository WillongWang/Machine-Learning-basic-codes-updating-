#!/usr/bin/env python3

# ============================
# Description: a guided grad_cam example code.
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

jet = matplotlib.colormaps["jet"]
jet_colors = jet(np.arange(256))[:, :3]

def plot_result(ax, img, saliency, label = ""):
    img = np.array(img, dtype = float) / 255.0

    saliency = F.interpolate(saliency, size = img.shape[:2], mode = "bilinear")
    saliency = as_numpy(saliency)[0, 0]
    saliency = saliency - saliency.min()
    saliency = np.uint8(255 * saliency / saliency.max())
    heatmap = jet_colors[saliency]
    ax.imshow(0.5 * heatmap + 0.5 * img)
    ax.axis("off")
    ax.set_title(label)

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
        [(*indx2label[str(i)], i, pred[i]) for i in pred.argsort()[::-1][:k]]
        for pred in as_numpy(preds)
    ]

class Probe:
    def get_hook(self,):
        self.data = []
        def hook(module, input, output):
            self.data.append(output)
        return hook

def Guided_ReLU_hook(m, g_i, g_o): # hook(module, grad_input, grad_output) -> tuple(Tensor) or None
    if isinstance(g_i, tuple):
        return tuple(g.clamp(min = 0) for g in g_i)
    return g_i.clamp(min = 0)

# load the image
print("loading the image...")
img = Image.open("./data/dog.png")
#img = Image.open("./data/1.png")
img = img.convert("RGB")


x = transform(img)[None]  # transform and reshape it to [1, C, *image_shape]
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
handle = model.layer4.register_forward_hook(probe.get_hook())

#using guided relu for backprop
handles = [handle]
for _, m in model.named_modules():
    if isinstance(m, torch.nn.ReLU):
        m.inplace = False # 为class torch.nn.ReLU(inplace=False)中的inplace
        handle = m.register_full_backward_hook(Guided_ReLU_hook) # 看看resnet.py
        handles.append(handle)

x = x.requires_grad_()
x.retain_grad()

logits = model(x)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
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
    # get the last_conv_output
    last_conv_output = probe.data[0]
    last_conv_output.retain_grad() #make sure the intermediate result save its grad

    #backprop
    logits[0, target].backward(retain_graph=True)
    grad = last_conv_output.grad 
    #taking average on the H-W panel
    weight = grad.mean(dim = (-1, -2), keepdim = True)
    saliency = (last_conv_output * weight).sum(dim = 1, keepdim = True) # (1, 1, H, W)
    #relu
    saliency = saliency.clamp(min = 0)
    guided_saliency = x.grad.abs().max(dim = 1, keepdim = True).values # x shape (1,C,H,W)
    guided_saliency *= F.interpolate(saliency, size = guided_saliency.shape[-2:], mode = "bilinear")
    
    last_conv_output.grad.zero_()
    x.grad.zero_()

    ax = plt.subplot(N, 1, i + 1)
    plot_result(ax, img, guided_saliency, "guided grad_cam on {} {}".format(*indx2label[str(target)]))

for handle in handles:
    handle.remove()
plt.savefig('output_image1.png')
