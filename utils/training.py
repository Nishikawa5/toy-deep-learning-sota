"""
Training utilities and helper functions.
"""

import torch
import torch.nn as nn
from tqdm import tqdm


def train_epoch(model, train_loader, optimizer, criterion, device='cuda'):
    """
    Train model for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        optimizer: Optimizer instance
        criterion: Loss function
        device: Device to train on
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(train_loader)


def evaluate(model, data_loader, criterion, device='cuda'):
    """
    Evaluate model on a dataset.
    
    Args:
        model: PyTorch model
        data_loader: Data loader (validation or test)
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        avg_loss, accuracy
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc='Evaluating'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model, path):
    """
    Save model weights (state_dict).
    
    Args:
        model: PyTorch model
        path: Path to save the model
    """
    import os
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path, device='cuda'):
    """
    Load model weights (state_dict).
    
    Args:
        model: PyTorch model
        path: Path to load the model from
        device: Device to load the model to
        
    Returns:
        Model with loaded weights
    """
    if str(device) == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"Model loaded from {path}")
    return model


def save_checkpoint(model, optimizer, epoch, train_loss, path):
    """
    Save training checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        train_loss: Current training loss
        path: Path to save the checkpoint
    """
    import os
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path, device='cuda'):
    """
    Load training checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        path: Path to load the checkpoint from
        device: Device to load the checkpoint to
        
    Returns:
        model, optimizer, epoch, loss
    """
    if str(device) == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['train_loss']
    
    model.to(device)
    
    # Move optimizer state to device
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
                
    print(f"Checkpoint loaded from {path} (Epoch {epoch}, Loss {loss:.4f})")
    return model, optimizer, epoch, loss
