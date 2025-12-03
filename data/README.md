# Data Directory

## Structure

- `raw/` - Original datasets (downloaded or generated)
- `processed/` - Preprocessed and cleaned data

## Available Datasets

### Vision
- **MNIST** - Handwritten digits (28x28 grayscale)
- **CIFAR-10** - Natural images (32x32 RGB, 10 classes)

### Text
- **WikiText-2** - Language modeling benchmark
- **Shakespeare** - Character-level text generation (Tiny Shakespeare)

### Time Series Forecasting

#### Synthetic
- **Sine Waves** - Multi-frequency synthetic signals
- **Multivariate TS** - Trend + seasonality patterns

#### Real-World
- **ETT (Electricity Transformer Temperature)** - `load_ett_dataset()`
  - Source: [ETDataset](https://github.com/zhouhaoyi/ETDataset)
  - Variants: ETTh1, ETTh2 (hourly), ETTm1, ETTm2 (15-min)
  - Features: Oil temperature, load data
  - Auto-downloads on first use
  
- **Air Quality UCI** - `load_air_quality_uci()`
  - Source: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Air+Quality)
  - Chemical sensors in Italian city
  - Features: CO, NOx, NO2, temperature, humidity
  - Auto-downloads on first use

### Anomaly Detection

#### Synthetic
- **Generic Anomalies** - Spikes, shifts, noise bursts, pattern changes
- **ECG Signals** - Synthetic heartbeats with arrhythmia

#### Real-World
- **KDD Cup 99** - `load_kddcup99()`
  - Source: [KDD Archive](http://kdd.ics.uci.edu/databases/kddcup99/)
  - Network intrusion detection (5M records)
  - Binary classification: normal vs attack
  - Auto-downloads on first use (~18MB)
  - Use `sample_size` parameter for faster testing

- **Credit Card Fraud** - `load_creditcard_fraud()`
  - Source: [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
  - **Manual download required** - place `creditcard.csv` in `data/raw/`
  - Highly imbalanced (0.172% fraud)
  - PCA-transformed features for privacy

## Adding New Datasets

1. Download raw data to `raw/` (auto or manual)
2. Use provided loaders from `utils.data_loaders`
3. For new datasets, add loader function and update this README

## Dataset Characteristics

| Dataset | Type | Size | Features | Auto-Download |
|---------|------|------|----------|---------------|
| MNIST | Vision | 60K train | 784 (28x28) | ✅ |
| CIFAR-10 | Vision | 50K train | 3x32x32 | ✅ |
| WikiText-2 | Text | 2M tokens | Variable | ✅ |
| Shakespeare | Text | 1MB | ~65 chars | ✅ |
| ETTh1 | Time Series | 17K samples | 7 features | ✅ |
| Air Quality | Time Series | 9K samples | ~10 features | ✅ |
| KDD Cup 99 | Anomaly | 5M records | 41 features | ✅ |
| Credit Card | Anomaly | 285K records | 30 features | ❌ Manual |
