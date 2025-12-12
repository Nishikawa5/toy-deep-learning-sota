"""
Utility functions for DL architecture experiments.
"""

from .data_loaders import (
    # Vision
    load_mnist, 
    load_cifar10,
    # Text
    load_wikitext2,
    load_shakespeare,
    # Time Series (Synthetic)
    generate_sine_wave_data,
    generate_multivariate_ts,
    # Time Series (Real)
    load_ett_dataset,
    load_air_quality_uci,
    # Anomaly Detection (Synthetic)
    generate_anomaly_data,
    generate_ecg_anomalies,
    # Anomaly Detection (Real)
    load_nsl_kdd,
    load_crypto_dataset,
    load_creditcard_fraud
)
from .training import train_epoch, evaluate, count_parameters, save_model, load_model, save_checkpoint, load_checkpoint
from .visualization import plot_training_history, visualize_predictions, plot_attention_weights, visualize_batch, visualize_reconstruction
from .metrics import compute_confusion_matrix, plot_confusion_matrix, get_classification_report

__all__ = [
    # Vision
    'load_mnist',
    'load_cifar10',
    # Text
    'load_wikitext2',
    'load_shakespeare',
    # Time Series (Synthetic)
    'generate_sine_wave_data',
    'generate_multivariate_ts',
    # Time Series (Real)
    'load_ett_dataset',
    'load_air_quality_uci',
    # Anomaly Detection (Synthetic)
    'generate_anomaly_data',
    'generate_ecg_anomalies',
    # Anomaly Detection (Real)
    'load_nsl_kdd',
    'load_crypto_dataset',
    'load_creditcard_fraud',
    # Training
    'train_epoch',
    'evaluate',
    'count_parameters',
    'save_model',
    'load_model',
    'save_checkpoint',
    'load_checkpoint',
    # Visualization
    'plot_training_history',
    'visualize_predictions',
    'plot_attention_weights',
    'visualize_batch',
    'visualize_reconstruction',
    # Metrics
    'compute_confusion_matrix',
    'plot_confusion_matrix',
    'get_classification_report',
]
