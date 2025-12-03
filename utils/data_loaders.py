"""
Common data loading utilities for all architectures.

Includes loaders for:
- Vision: MNIST, CIFAR-10
- Text: WikiText-2, IMDB sentiment
- Time Series: Synthetic forecasting, electricity load
- Anomaly Detection: Synthetic anomalies, ECG data
"""

import torch
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import os


def get_data_dir():
    """Get the path to the data directory."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


# ============================================================================
# VISION DATASETS
# ============================================================================

def load_mnist(batch_size=64, val_split=0.1):
    """
    Load MNIST dataset with train/val split.
    
    Args:
        batch_size: Batch size for DataLoader
        val_split: Fraction of training data to use for validation
        
    Returns:
        train_loader, val_loader, test_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    data_dir = os.path.join(get_data_dir(), 'raw')
    
    # Download/load datasets
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    
    # Split training into train/val
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def load_cifar10(batch_size=64, val_split=0.1):
    """
    Load CIFAR-10 dataset with train/val split.
    
    Args:
        batch_size: Batch size for DataLoader
        val_split: Fraction of training data to use for validation
        
    Returns:
        train_loader, val_loader, test_loader
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    data_dir = os.path.join(get_data_dir(), 'raw')
    
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)
    
    # Split training into train/val
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader


# ============================================================================
# TEXT DATASETS
# ============================================================================

class TextDataset(Dataset):
    """Simple text dataset for character or word-level language modeling."""
    
    def __init__(self, text, seq_length, char_level=True):
        """
        Args:
            text: String of text data
            seq_length: Length of sequences to generate
            char_level: If True, use character-level; else word-level
        """
        self.seq_length = seq_length
        self.char_level = char_level
        
        if char_level:
            self.chars = sorted(list(set(text)))
            self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
            self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
            self.vocab_size = len(self.chars)
            self.data = [self.char_to_idx[ch] for ch in text]
        else:
            words = text.split()
            self.vocab = sorted(list(set(words)))
            self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
            self.idx_to_word = {i: w for i, w in enumerate(self.vocab)}
            self.vocab_size = len(self.vocab)
            self.data = [self.word_to_idx[w] for w in words]
    
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx + self.seq_length], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + self.seq_length + 1], dtype=torch.long)
        return x, y


def load_wikitext2(batch_size=32, seq_length=100, val_split=0.1):
    """
    Load WikiText-2 dataset for language modeling.
    
    Args:
        batch_size: Batch size
        seq_length: Sequence length for training
        val_split: Validation split ratio
        
    Returns:
        train_loader, val_loader, test_loader, vocab_size
    """
    try:
        from torchtext.datasets import WikiText2
        from torchtext.data.utils import get_tokenizer
    except ImportError:
        raise ImportError("torchtext required. Install with: pip install torchtext")
    
    data_dir = os.path.join(get_data_dir(), 'raw')
    
    # Load dataset
    train_iter, val_iter, test_iter = WikiText2(root=data_dir)
    
    # Build vocabulary
    tokenizer = get_tokenizer('basic_english')
    vocab = {}
    idx = 0
    
    def build_vocab(data_iter):
        nonlocal idx
        for line in data_iter:
            tokens = tokenizer(line)
            for token in tokens:
                if token not in vocab:
                    vocab[token] = idx
                    idx += 1
    
    # Build vocab from train set
    build_vocab(train_iter)
    vocab_size = len(vocab)
    
    # Convert to indices
    def text_to_indices(data_iter):
        data = []
        for line in data_iter:
            tokens = tokenizer(line)
            data.extend([vocab.get(token, 0) for token in tokens])
        return data
    
    # Recreate iterators (they're consumed after building vocab)
    train_iter, val_iter, test_iter = WikiText2(root=data_dir)
    
    train_data = text_to_indices(train_iter)
    val_data = text_to_indices(val_iter)
    test_data = text_to_indices(test_iter)
    
    # Create datasets
    class SequenceDataset(Dataset):
        def __init__(self, data, seq_length):
            self.data = data
            self.seq_length = seq_length
        
        def __len__(self):
            return len(self.data) - self.seq_length
        
        def __getitem__(self, idx):
            x = torch.tensor(self.data[idx:idx + self.seq_length], dtype=torch.long)
            y = torch.tensor(self.data[idx + 1:idx + self.seq_length + 1], dtype=torch.long)
            return x, y
    
    train_dataset = SequenceDataset(train_data, seq_length)
    val_dataset = SequenceDataset(val_data, seq_length)
    test_dataset = SequenceDataset(test_data, seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, vocab_size


def load_shakespeare(batch_size=64, seq_length=100):
    """
    Load Shakespeare text dataset for character-level language modeling.
    Downloads from a public source.
    
    Args:
        batch_size: Batch size
        seq_length: Sequence length
        
    Returns:
        train_loader, val_loader, vocab_size, char_to_idx, idx_to_char
    """
    import requests
    
    data_dir = os.path.join(get_data_dir(), 'raw')
    os.makedirs(data_dir, exist_ok=True)
    
    file_path = os.path.join(data_dir, 'shakespeare.txt')
    
    # Download if not exists
    if not os.path.exists(file_path):
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        response = requests.get(url)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
    
    # Load text
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Create dataset
    dataset = TextDataset(text, seq_length, char_level=True)
    
    # Split
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, dataset.vocab_size, dataset.char_to_idx, dataset.idx_to_char


# ============================================================================
# TIME SERIES FORECASTING
# ============================================================================

def generate_sine_wave_data(num_samples=10000, seq_length=50, pred_length=10, 
                           num_frequencies=3, noise_level=0.1):
    """
    Generate synthetic sine wave data for time series forecasting.
    
    Args:
        num_samples: Number of sequences to generate
        seq_length: Length of input sequence
        pred_length: Length of prediction horizon
        num_frequencies: Number of sine waves to combine
        noise_level: Amount of noise to add
        
    Returns:
        train_loader, val_loader, test_loader
    """
    np.random.seed(42)
    
    all_sequences = []
    all_targets = []
    
    for _ in range(num_samples):
        # Generate random frequencies and phases
        frequencies = np.random.uniform(0.5, 5, num_frequencies)
        phases = np.random.uniform(0, 2 * np.pi, num_frequencies)
        amplitudes = np.random.uniform(0.5, 2, num_frequencies)
        
        # Generate time points
        t = np.linspace(0, 10, seq_length + pred_length)
        
        # Combine multiple sine waves
        signal = np.zeros_like(t)
        for freq, phase, amp in zip(frequencies, phases, amplitudes):
            signal += amp * np.sin(2 * np.pi * freq * t + phase)
        
        # Add trend
        trend = np.linspace(0, np.random.uniform(-1, 1), len(t))
        signal += trend
        
        # Add noise
        signal += np.random.normal(0, noise_level, len(t))
        
        # Split into input and target
        x = signal[:seq_length]
        y = signal[seq_length:seq_length + pred_length]
        
        all_sequences.append(x)
        all_targets.append(y)
    
    # Convert to tensors
    X = torch.FloatTensor(all_sequences).unsqueeze(-1)  # (num_samples, seq_length, 1)
    Y = torch.FloatTensor(all_targets).unsqueeze(-1)    # (num_samples, pred_length, 1)
    
    # Create dataset
    dataset = TensorDataset(X, Y)
    
    # Split
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, test_loader


def generate_multivariate_ts(num_samples=5000, seq_length=100, pred_length=20, 
                             num_features=5, seasonality=24):
    """
    Generate multivariate time series with trend, seasonality, and noise.
    
    Args:
        num_samples: Number of sequences
        seq_length: Input sequence length
        pred_length: Prediction horizon
        num_features: Number of features/variables
        seasonality: Seasonal period
        
    Returns:
        train_loader, val_loader, test_loader
    """
    np.random.seed(42)
    
    all_sequences = []
    all_targets = []
    
    for _ in range(num_samples):
        t = np.arange(seq_length + pred_length)
        data = np.zeros((seq_length + pred_length, num_features))
        
        for i in range(num_features):
            # Trend
            trend = np.linspace(0, np.random.uniform(-2, 2), len(t))
            
            # Seasonality
            seasonal = np.sin(2 * np.pi * t / seasonality + np.random.uniform(0, 2*np.pi))
            
            # Random walk component
            random_walk = np.cumsum(np.random.normal(0, 0.1, len(t)))
            
            # Combine
            data[:, i] = trend + seasonal + random_walk + np.random.normal(0, 0.1, len(t))
        
        # Split
        x = data[:seq_length]
        y = data[seq_length:seq_length + pred_length]
        
        all_sequences.append(x)
        all_targets.append(y)
    
    X = torch.FloatTensor(all_sequences)
    Y = torch.FloatTensor(all_targets)
    
    dataset = TensorDataset(X, Y)
    
    # Split
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, test_loader


# ============================================================================
# ANOMALY DETECTION
# ============================================================================

def generate_anomaly_data(num_normal=8000, num_anomalies=500, seq_length=100, 
                         num_features=5):
    """
    Generate synthetic time series data with anomalies.
    
    Args:
        num_normal: Number of normal sequences
        num_anomalies: Number of anomalous sequences
        seq_length: Sequence length
        num_features: Number of features
        
    Returns:
        train_loader (normal only), test_loader (mixed), labels
    """
    np.random.seed(42)
    
    def generate_normal_sequence():
        """Generate a normal sequence with typical patterns."""
        t = np.linspace(0, 10, seq_length)
        data = np.zeros((seq_length, num_features))
        
        for i in range(num_features):
            # Smooth sine wave with small noise
            freq = np.random.uniform(0.5, 2)
            phase = np.random.uniform(0, 2 * np.pi)
            data[:, i] = np.sin(2 * np.pi * freq * t + phase)
            data[:, i] += np.random.normal(0, 0.1, seq_length)
        
        return data
    
    def generate_anomaly_sequence():
        """Generate an anomalous sequence."""
        data = generate_normal_sequence()
        
        # Add different types of anomalies
        anomaly_type = np.random.choice(['spike', 'shift', 'noise', 'pattern_change'])
        
        if anomaly_type == 'spike':
            # Add random spikes
            num_spikes = np.random.randint(1, 5)
            for _ in range(num_spikes):
                idx = np.random.randint(0, seq_length)
                feature = np.random.randint(0, num_features)
                data[idx, feature] += np.random.uniform(3, 5) * np.random.choice([-1, 1])
        
        elif anomaly_type == 'shift':
            # Level shift
            shift_point = np.random.randint(seq_length // 2, seq_length)
            feature = np.random.randint(0, num_features)
            data[shift_point:, feature] += np.random.uniform(2, 4)
        
        elif anomaly_type == 'noise':
            # Increased noise level
            feature = np.random.randint(0, num_features)
            data[:, feature] += np.random.normal(0, 1, seq_length)
        
        else:  # pattern_change
            # Change frequency in second half
            change_point = seq_length // 2
            feature = np.random.randint(0, num_features)
            t = np.linspace(0, 5, seq_length - change_point)
            data[change_point:, feature] = np.sin(2 * np.pi * 10 * t)
        
        return data
    
    # Generate data
    normal_data = [generate_normal_sequence() for _ in range(num_normal)]
    anomaly_data = [generate_anomaly_sequence() for _ in range(num_anomalies)]
    
    # Training data (normal only)
    X_train = torch.FloatTensor(normal_data)
    train_dataset = TensorDataset(X_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Test data (mixed)
    X_test = torch.FloatTensor(normal_data[-1000:] + anomaly_data)
    y_test = torch.cat([
        torch.zeros(1000),  # Normal
        torch.ones(num_anomalies)  # Anomaly
    ])
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader


def generate_ecg_anomalies(num_normal=5000, num_anomalies=300, seq_length=200):
    """
    Generate synthetic ECG-like signals with anomalies.
    
    Args:
        num_normal: Number of normal heartbeat sequences
        num_anomalies: Number of anomalous heartbeats
        seq_length: Sequence length
        
    Returns:
        train_loader, test_loader
    """
    np.random.seed(42)
    
    def generate_normal_heartbeat():
        """Generate a normal ECG heartbeat pattern."""
        t = np.linspace(0, 1, seq_length)
        
        # P wave
        p_wave = 0.25 * np.exp(-((t - 0.25)**2) / 0.001)
        
        # QRS complex
        q_wave = -0.1 * np.exp(-((t - 0.45)**2) / 0.0001)
        r_wave = 1.5 * np.exp(-((t - 0.5)**2) / 0.0001)
        s_wave = -0.3 * np.exp(-((t - 0.55)**2) / 0.0001)
        
        # T wave
        t_wave = 0.4 * np.exp(-((t - 0.75)**2) / 0.002)
        
        ecg = p_wave + q_wave + r_wave + s_wave + t_wave
        ecg += np.random.normal(0, 0.02, seq_length)  # Add noise
        
        return ecg
    
    def generate_anomalous_heartbeat():
        """Generate an anomalous ECG pattern."""
        ecg = generate_normal_heartbeat()
        
        anomaly_type = np.random.choice(['arrhythmia', 'noise', 'missing_wave'])
        
        if anomaly_type == 'arrhythmia':
            # Irregular rhythm - stretch or compress
            factor = np.random.uniform(0.7, 1.5)
            t_new = np.linspace(0, factor, seq_length)
            ecg = np.interp(t_new, np.linspace(0, 1, seq_length), ecg)
        
        elif anomaly_type == 'noise':
            # High frequency noise
            ecg += np.random.normal(0, 0.2, seq_length)
        
        else:  # missing_wave
            # Suppress T wave or P wave
            if np.random.random() > 0.5:
                # Remove T wave region
                mask = (np.arange(seq_length) > 0.65 * seq_length) & \
                       (np.arange(seq_length) < 0.85 * seq_length)
                ecg[mask] *= 0.2
        
        return ecg
    
    # Generate data
    normal_ecg = np.array([generate_normal_heartbeat() for _ in range(num_normal)])
    anomaly_ecg = np.array([generate_anomalous_heartbeat() for _ in range(num_anomalies)])
    
    # Training (normal only)
    X_train = torch.FloatTensor(normal_ecg).unsqueeze(-1)
    train_dataset = TensorDataset(X_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Test (mixed)
    X_test = torch.FloatTensor(np.concatenate([normal_ecg[-500:], anomaly_ecg])).unsqueeze(-1)
    y_test = torch.cat([torch.zeros(500), torch.ones(num_anomalies)])
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader


# ============================================================================
# REAL-WORLD TIME SERIES DATASETS
# ============================================================================

def load_ett_dataset(dataset_name='ETTh1', seq_length=96, pred_length=24, batch_size=32):
    """
    Load Electricity Transformer Temperature (ETT) dataset.
    
    Widely used benchmark for time series forecasting. Contains:
    - Oil temperature
    - Load features
    - Hourly (ETTh1, ETTh2) or 15-min (ETTm1, ETTm2) intervals
    
    Args:
        dataset_name: One of ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']
        seq_length: Input sequence length
        pred_length: Prediction horizon
        batch_size: Batch size
        
    Returns:
        train_loader, val_loader, test_loader, num_features
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    
    data_dir = os.path.join(get_data_dir(), 'raw', 'ETT-small')
    os.makedirs(data_dir, exist_ok=True)
    
    file_path = os.path.join(data_dir, f'{dataset_name}.csv')
    
    # Download if not exists
    if not os.path.exists(file_path):
        import requests
        url = f'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{dataset_name}.csv'
        print(f'Downloading {dataset_name} dataset...')
        response = requests.get(url)
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print('Download complete!')
    
    # Load data
    df = pd.read_csv(file_path)
    df = df.drop(columns=['date'])  # Remove date column
    
    # Split into train/val/test (70/10/20 split as per standard)
    num_samples = len(df)
    train_end = int(0.7 * num_samples)
    val_end = int(0.8 * num_samples)
    
    train_data = df[:train_end].values
    val_data = df[train_end:val_end].values
    test_data = df[val_end:].values
    
    # Normalize using train statistics
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    val_data = scaler.transform(val_data)
    test_data = scaler.transform(test_data)
    
    # Create sequences
    def create_sequences(data, seq_len, pred_len):
        X, Y = [], []
        for i in range(len(data) - seq_len - pred_len + 1):
            X.append(data[i:i+seq_len])
            Y.append(data[i+seq_len:i+seq_len+pred_len])
        return np.array(X), np.array(Y)
    
    X_train, Y_train = create_sequences(train_data, seq_length, pred_length)
    X_val, Y_val = create_sequences(val_data, seq_length, pred_length)
    X_test, Y_test = create_sequences(test_data, seq_length, pred_length)
    
    # Convert to tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(Y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(Y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(Y_test)
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    num_features = train_data.shape[1]
    
    return train_loader, val_loader, test_loader, num_features


def load_air_quality_uci(seq_length=48, pred_length=12, batch_size=32):
    """
    Load Air Quality dataset from UCI.
    
    Contains hourly averaged responses from chemical sensors in an Italian city.
    Features: CO, NOx, NO2, temperature, humidity, etc.
    
    Args:
        seq_length: Input sequence length (hours)
        pred_length: Prediction horizon (hours)
        batch_size: Batch size
        
    Returns:
        train_loader, val_loader, test_loader, num_features
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    
    data_dir = os.path.join(get_data_dir(), 'raw')
    os.makedirs(data_dir, exist_ok=True)
    
    file_path = os.path.join(data_dir, 'AirQualityUCI.csv')
    
    # Download if not exists
    if not os.path.exists(file_path):
        import zipfile
        import requests
        
        print('Downloading Air Quality UCI dataset...')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip'
        zip_path = os.path.join(data_dir, 'AirQualityUCI.zip')
        
        response = requests.get(url)
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        
        # Extract
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        os.remove(zip_path)
        print('Download complete!')
    
    # Load data
    df = pd.read_csv(file_path, sep=';', decimal=',')
    
    # Keep only numeric columns and handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df[numeric_cols]
    
    # Replace -200 (missing value indicator) with NaN and forward fill
    df = df.replace(-200, np.nan)
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Remove any remaining NaN rows
    df = df.dropna()
    
    data = df.values
    
    # Split
    num_samples = len(data)
    train_end = int(0.7 * num_samples)
    val_end = int(0.8 * num_samples)
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    # Normalize
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    val_data = scaler.transform(val_data)
    test_data = scaler.transform(test_data)
    
    # Create sequences
    def create_sequences(data, seq_len, pred_len):
        X, Y = [], []
        for i in range(len(data) - seq_len - pred_len + 1):
            X.append(data[i:i+seq_len])
            Y.append(data[i+seq_len:i+seq_len+pred_len])
        return np.array(X), np.array(Y)
    
    X_train, Y_train = create_sequences(train_data, seq_length, pred_length)
    X_val, Y_val = create_sequences(val_data, seq_length, pred_length)
    X_test, Y_test = create_sequences(test_data, seq_length, pred_length)
    
    # Convert to tensors
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(Y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(Y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    num_features = train_data.shape[1]
    
    return train_loader, val_loader, test_loader, num_features


# ============================================================================
# REAL-WORLD ANOMALY DETECTION DATASETS
# ============================================================================

def load_kddcup99(sample_size=None, batch_size=64):
    """
    Load KDD Cup 99 network intrusion detection dataset.
    
    Classic anomaly detection benchmark. Contains network connection records
    labeled as normal or various types of attacks.
    
    Args:
        sample_size: If specified, randomly sample this many records (for faster testing)
        batch_size: Batch size
        
    Returns:
        train_loader, test_loader, feature_names
    """
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    
    data_dir = os.path.join(get_data_dir(), 'raw')
    os.makedirs(data_dir, exist_ok=True)
    
    file_path = os.path.join(data_dir, 'kddcup.data.gz')
    
    # Download if not exists
    if not os.path.exists(file_path):
        import requests
        print('Downloading KDD Cup 99 dataset (takes a few minutes)...')
        url = 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz'
        response = requests.get(url, stream=True)
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print('Download complete!')
    
    # Column names
    column_names = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
    ]
    
    # Load data
    df = pd.read_csv(file_path, names=column_names, compression='gzip')
    
    # Sample if requested
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
    
    # Binary labels: normal vs attack
    df['is_attack'] = (df['label'] != 'normal.').astype(int)
    
    # Separate features and labels
    y = df['is_attack'].values
    X = df.drop(columns=['label', 'is_attack'])
    
    # Encode categorical features
    categorical_cols = ['protocol_type', 'service', 'flag']
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    X = X.values.astype(np.float32)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    feature_names = list(X.columns) if hasattr(X, 'columns') else column_names[:-1]
    
    return train_loader, test_loader, feature_names


def load_creditcard_fraud(sample_size=None, batch_size=64):
    """
    Load Credit Card Fraud Detection dataset (Kaggle).
    
    Highly imbalanced dataset with credit card transactions.
    Features are PCA-transformed for privacy.
    
    Note: You need to download this manually from:
    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    Place creditcard.csv in data/raw/
    
    Args:
        sample_size: If specified, sample this many records
        batch_size: Batch size
        
    Returns:
        train_loader, test_loader, num_features
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    
    data_dir = os.path.join(get_data_dir(), 'raw')
    file_path = os.path.join(data_dir, 'creditcard.csv')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Credit Card dataset not found at {file_path}\n"
            "Please download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "Place creditcard.csv in data/raw/"
        )
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Sample if requested
    if sample_size and sample_size < len(df):
        # Stratified sampling to maintain fraud ratio
        normal = df[df['Class'] == 0].sample(n=int(sample_size * 0.998), random_state=42)
        fraud = df[df['Class'] == 1].sample(n=int(sample_size * 0.002), random_state=42)
        df = pd.concat([normal, fraud]).sample(frac=1, random_state=42)
    
    # Separate features and labels
    y = df['Class'].values
    X = df.drop(columns=['Class', 'Time']).values  # Remove Time and Class
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Normalize (Amount column varies widely)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    num_features = X_train.shape[1]
    
    print(f"Dataset loaded: {len(X_train)} train samples, {len(X_test)} test samples")
    print(f"Fraud ratio - Train: {y_train.mean():.4f}, Test: {y_test.mean():.4f}")
    
    return train_loader, test_loader, num_features
