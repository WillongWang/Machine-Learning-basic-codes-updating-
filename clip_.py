import numpy as np
import torch
import random
from pkg_resources import packaging

print("Torch version:", torch.__version__)

import clip  # pip install openai-clip

clip.available_models()

model, preprocess = clip.load("ViT-B/16")

model.cuda().eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

import os
import IPython.display
import matplotlib.pyplot as plt
from PIL import Image

from collections import OrderedDict

from torchvision.datasets import Caltech101

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'


caltech101 = Caltech101(root="/root", transform=preprocess, download=False)


def accuracy(predicted, labels):
    predictions, id = torch.max(predicted, dim=1)
    return torch.tensor(torch.sum(id == labels).item() / len(predicted))


print(len(caltech101.categories))
print(caltech101.categories)
text_descriptions = [f"a photo of a {label}" for label in caltech101.categories]
text_descriptions1 = [f"a photo of the {label}" for label in caltech101.categories]
text_descriptions2 = [f" 'a painting of a {label}.'" for label in caltech101.categories]
text_descriptions3 = [f" 'a painting of the {label}.'" for label in caltech101.categories]
text_tokens_1 = clip.tokenize(text_descriptions).cuda()
text_tokens_2 = clip.tokenize(text_descriptions1).cuda()
text_tokens_3 = clip.tokenize(text_descriptions2).cuda()
text_tokens_4 = clip.tokenize(text_descriptions3).cuda()

print(text_tokens_1.shape)
with torch.no_grad():
    text_features_1 = model.encode_text(text_tokens_1).float()
    text_features_2 = model.encode_text(text_tokens_2).float()
    text_features_3 = model.encode_text(text_tokens_3).float()
    text_features_4 = model.encode_text(text_tokens_4).float()

    # Normalize the embeddings
    text_features_1 /= text_features_1.norm(dim=-1, keepdim=True)
    text_features_2 /= text_features_2.norm(dim=-1, keepdim=True)
    text_features_3 /= text_features_3.norm(dim=-1, keepdim=True)
    text_features_4 /= text_features_4.norm(dim=-1, keepdim=True)

    # Average the embeddings
    text_features = (text_features_1 + text_features_2 + text_features_3 + text_features_4) / 4
print(text_features.shape)

caltech101_testset = Caltech101(root="/root", download=False, transform=preprocess)
caltech101_test_loader = torch.utils.data.DataLoader(caltech101_testset, batch_size=2000, shuffle=False,
                                                     pin_memory=True)
print(len(caltech101_test_loader))

accuracy_per_batch = []
current_epoch = 1

misclassified_images = []
misclassified_labels = []
misclassified_predictions = []

for images, labels in caltech101_test_loader:
    print(str(current_epoch) + '/' + str(len(caltech101_test_loader)))
    with torch.no_grad():
        images = images.cuda()
        labels = labels.cuda()
        # get the image embeddings
        image_features = model.encode_image(images).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        print('image_features.shape:' + str(image_features.shape))
        print('text_features.shape:' + str(text_features.shape))
        # multiple the image embeddings with the clasificication head to get the predicted logits
        # [2000, 512] [512, 100] --> [2000, 100]
        # 2000: number of images, 512:embedding size, 100: number of classes
        pred_logits = 100.0 * image_features @ text_features.T
        print('pred_logits.shape:' + str(pred_logits.shape))
        pred_probs = pred_logits.softmax(dim=-1)
        acc_batch = accuracy(pred_probs, labels)
        print('accuracy at this batch:' + str(acc_batch.item() * 100) + '%')
        accuracy_per_batch.append(acc_batch)

        predicted_classes = pred_probs.argmax(dim=-1)
        misclassified = (predicted_classes != labels)

        misclassified_images.extend(images[misclassified].cpu())
        misclassified_labels.extend(labels[misclassified].cpu())
        misclassified_predictions.extend(predicted_classes[misclassified].cpu())

    current_epoch += 1

overall_accuracy = torch.stack(accuracy_per_batch).mean().item()
print('The zero-shot prediction accuracy on the caltech101 testing set:')
print(str(overall_accuracy * 100) + '%')

num_images_to_display = min(10, len(misclassified_images))  # Display up to 10 images
plt.figure(figsize=(16, 5))

random_indices = random.sample(range(len(misclassified_images)), num_images_to_display)

for i, ii in enumerate(random_indices):
    image = misclassified_images[ii].permute(1, 2, 0).numpy()  # ？
    image = (image - image.min()) / (image.max() - image.min())
    true_label = caltech101.categories[misclassified_labels[ii]]
    predicted_label = caltech101.categories[misclassified_predictions[ii]]

    plt.subplot(2, 5, i + 1)
    plt.imshow(image)
    plt.title(f"True: {true_label}\nPred: {predicted_label}")
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
# plt.show(block=True)
