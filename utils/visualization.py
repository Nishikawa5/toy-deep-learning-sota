"""
Visualization utilities for experiments.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_training_history(train_losses, val_losses, val_accuracies, save_path=None):
    """
    Plot training history.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        val_accuracies: List of validation accuracies per epoch
        save_path: Optional path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_predictions(model, data_loader, device='cuda', num_samples=10):
    """
    Visualize model predictions on sample data.
    
    Args:
        model: Trained model
        data_loader: Data loader to sample from
        device: Device to run inference on
        num_samples: Number of samples to visualize
    """
    model.eval()
    
    # Get a batch of data
    data, target = next(iter(data_loader))
    data, target = data.to(device), target.to(device)
    
    with torch.no_grad():
        output = model(data)
        pred = output.argmax(dim=1)
    
    # Plot samples
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for idx in range(min(num_samples, len(data))):
        img = data[idx].cpu().squeeze()
        axes[idx].imshow(img, cmap='gray')
        axes[idx].axis('off')
        
        true_label = target[idx].item()
        pred_label = pred[idx].item()
        color = 'green' if true_label == pred_label else 'red'
        
        axes[idx].set_title(f'True: {true_label}\nPred: {pred_label}', 
                           color=color, fontsize=10)
    
    plt.tight_layout()
    plt.show()


def plot_attention_weights(attention_weights, save_path=None):
    """
    Visualize attention weights as a heatmap.
    
    Args:
        attention_weights: Tensor of shape (seq_len, seq_len) or (num_heads, seq_len, seq_len)
        save_path: Optional path to save the figure
    """
    if attention_weights.dim() == 3:
        # Multiple heads
        num_heads = attention_weights.shape[0]
        fig, axes = plt.subplots(1, num_heads, figsize=(5*num_heads, 4))
        if num_heads == 1:
            axes = [axes]
        
        for idx, ax in enumerate(axes):
            im = ax.imshow(attention_weights[idx].cpu().numpy(), cmap='viridis')
            ax.set_title(f'Head {idx+1}')
            ax.set_xlabel('Key')
            ax.set_ylabel('Query')
            plt.colorbar(im, ax=ax)
    else:
        # Single attention matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(attention_weights.cpu().numpy(), cmap='viridis')
        ax.set_title('Attention Weights')
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')
        plt.colorbar(im)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
