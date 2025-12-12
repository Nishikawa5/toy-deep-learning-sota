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


def visualize_batch(images, nrow=8, title=None, save_path=None, denormalize=True, show=True):
    """
    Visualize a batch of images in a grid.
    
    Args:
        images: Tensor of shape (B, C, H, W)
        nrow: Number of images per row
        title: Optional title for the plot
        save_path: Optional path to save the figure
        denormalize: If True, scales [-1, 1] data to [0, 1]
    """
    if denormalize:
        images = (images + 1) / 2.0
        images = torch.clamp(images, 0, 1)
        
    # Convert to specific format for plotting
    if images.dim() == 4:
        # If single channel (B, 1, H, W) -> (B, H, W)
        if images.shape[1] == 1:
            images = images.squeeze(1)
        # If RGB (B, 3, H, W) -> (B, H, W, 3)
        elif images.shape[1] == 3:
            images = images.permute(0, 2, 3, 1)
            
    images = images.cpu().detach().numpy()
    
    # Calculate grid dimensions
    batch_size = len(images)
    n_rows = (batch_size + nrow - 1) // nrow
    
    fig, axes = plt.subplots(n_rows, nrow, figsize=(nrow * 1.5, n_rows * 1.5))
    if title:
        fig.suptitle(title)
        
    # Handle single row case
    if n_rows == 1:
        if nrow == 1:
             axes = np.array([axes])
        else:
             axes = axes.reshape(1, -1)
    elif nrow == 1:
        axes = axes.reshape(-1, 1)
        
    axes = axes.ravel()
    
    for i in range(len(axes)):
        if i < batch_size:
            if images[i].ndim == 2: # Grayscale
                axes[i].imshow(images[i], cmap='gray')
            else: # RGB
                axes[i].imshow(images[i])
            axes[i].axis('off')
        else:
            axes[i].axis('off')
            
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()


def visualize_reconstruction(real_images, recon_images, num_samples=5, save_path=None, denormalize=True):
    """
    Visualize original vs reconstructed images side by side.
    
    Args:
        real_images: Batch of original images
        recon_images: Batch of reconstructed images
        num_samples: Number of samples to visualize
        save_path: Optional path to save the figure
        denormalize: If True, scales [-1, 1] data to [0, 1]
    """
    # Helper to process images
    def process_imgs(imgs):
        if denormalize:
            imgs = (imgs + 1) / 2.0
            imgs = torch.clamp(imgs, 0, 1)
        return imgs.cpu().detach()

    real_images = process_imgs(real_images)
    recon_images = process_imgs(recon_images)
    
    num_samples = min(num_samples, len(real_images))
    
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
    
    # Handle case where num_samples is 1 (axes is 1D array)
    if num_samples == 1:
        axes = axes.reshape(2, 1)

    for i in range(num_samples):
        # Original
        if real_images.shape[1] == 1:
            img_real = real_images[i].squeeze()
            axes[0, i].imshow(img_real, cmap='gray')
        else:
            img_real = real_images[i].permute(1, 2, 0)
            axes[0, i].imshow(img_real)
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')
        
        # Reconstructed
        if recon_images.shape[1] == 1:
            img_recon = recon_images[i].squeeze()
            axes[1, i].imshow(img_recon, cmap='gray')
        else:
            img_recon = recon_images[i].permute(1, 2, 0)
            axes[1, i].imshow(img_recon)
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis('off')
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()

