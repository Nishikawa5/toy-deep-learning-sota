# Deep Learning Architectures Study

This repository is a collection of toy implementations of State-of-the-Art (SOTA) deep learning architectures.

The primary goal of this project is education. I am implementing these architectures from scratch to study the underlying mathematics, design decisions, and mechanics of how these architectures work. My aim is to bridge the gap between theory and code, making the abstract math of the architectures clearer by building the structures myself. I found this approach to be very helpful in understanding the architectures better.

While I may test these models on simple datasets, these experiments serve as verification tools to understand the how and why behind the architectures, rather than for performance benchmarking.

Note: These implementations are simplified for readability and my understanding. They are not optimized for production performance, training speed, or massive scale. They are just for learning purposes.


## Project Structure

```
dl-architectures-study/
├── data/                 # Datasets
│   ├── raw/              # Original data
│   └── processed/        # Preprocessed data
├── architectures/        # Architecture implementations
│   ├── transformers/
│   ├── recursive_models/
│   └── ssm/
├── utils/                # Shared utilities
├── configs/              # Experiment configurations
└── results/              # Outputs and checkpoints
```