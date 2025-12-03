"""
Evaluation metrics and analysis utilities.
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def compute_confusion_matrix(model, data_loader, num_classes, device='cuda'):
    """
    Compute confusion matrix for model predictions.
    
    Args:
        model: Trained model
        data_loader: Data loader
        num_classes: Number of classes
        device: Device to run on
        
    Returns:
        Confusion matrix as numpy array
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.numpy())
    
    return confusion_matrix(all_targets, all_preds, labels=range(num_classes))


def plot_confusion_matrix(cm, class_names=None, save_path=None):
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names (optional)
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(10, 8))
    
    if class_names is None:
        class_names = [str(i) for i in range(len(cm))]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def get_classification_report(model, data_loader, class_names=None, device='cuda'):
    """
    Generate classification report with precision, recall, F1.
    
    Args:
        model: Trained model
        data_loader: Data loader
        class_names: List of class names (optional)
        device: Device to run on
        
    Returns:
        Classification report string
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.numpy())
    
    return classification_report(all_targets, all_preds, target_names=class_names)
