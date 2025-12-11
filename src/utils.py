import torch
import matplotlib.pyplot as plt
import numpy as np

def set_device(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def plot_loss_curve(loss_list, title="Training Loss"):
    plt.figure(figsize=(6,4))
    plt.plot(loss_list, marker='o')
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

def show_images(images, n=6, cmap='gray'):
    plt.figure(figsize=(12,2))
    for i in range(n):
        plt.subplot(1,n,i+1)
        plt.imshow(images[i].cpu().squeeze(), cmap=cmap)
        plt.axis('off')
    plt.show()
