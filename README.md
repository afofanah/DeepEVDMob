# DeepEVD: Deep Learning for Epidemiological Forecasting
[Overall_Architecture.pdf](https://github.com/user-attachments/files/21705244/Overall_Architecture.pdf)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

DeepEVD is a state-of-the-art deep learning framework for spatio-temporal epidemiological forecasting, specifically designed for predicting disease transmission patterns in outbreak scenarios. The model integrates Graph Convolutional Networks (GCN) and Long Short-Term Memory (LSTM) networks to capture complex spatial dependencies and temporal dynamics in disease spread.


- Spatio-Temporal Modeling: Combines spatial graph structures with temporal sequences for comprehensive disease transmission modeling
- Multi-Task Learning: Simultaneously predicts case numbers, reproduction rates (R₀), and spatial risk maps
- Attention Mechanisms: Incorporates spatial and temporal attention for improved feature representation
- Flexible Architecture: Modular design supporting various epidemiological scenarios
- Real-Time Prediction: Supports both batch and streaming prediction modes

### Applications

- Outbreak Response: Early warning systems for disease outbreaks
- Resource Allocation: Optimizing healthcare resource distribution
- Policy Planning: Supporting evidence-based public health interventions
- Risk Assessment: Spatial risk mapping for vulnerable populations

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional but recommended)

### Dependencies

```bash
# Clone the repository
git clone https://github.com/username/deepevd.git
cd deepevd

# Create virtual environment
python -m venv deepevd_env
source deepevd_env/bin/activate  

# Install required packages
pip install -r requirements.txt
```

### Requirements.txt

```
torch>=1.9.0
torch-geometric>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
tqdm>=4.62.0
wandb>=0.12.0
networkx>=2.6.0
scipy>=1.7.0
```

## Model Architecture

### Core Components

1. Embedding Layer: Transforms sparse mobility data into dense vector representations using location co-occurrence matrix factorization

2. Spatial Feature Representation and Extraction (SFRE):
   - Graph Convolutional Networks with attention mechanisms
   - Captures spatial dependencies between geographic locations
   - Normalized adjacency matrices for stable training

3. Temporal Feature Representation and Extraction (TFRE):
   - Multi-layer LSTM with custom cells
   - Dilated convolutions for multi-scale temporal patterns
   - Temporal attention mechanisms

4. Prediction Layer:
   - Multi-head architecture for different prediction tasks
   - Case prediction (classification)
   - R₀ estimation (regression)
   - Risk mapping (binary classification)

### Model Configuration

```python
config = {
    'model': {
        'input_dim': 8,           # Input feature dimensions
        'embedding_dim': 32,      # Embedding layer output dimension
        'spatial_dim': 64,        # Spatial feature dimension
        'temporal_dim': 128,      # Temporal feature dimension
        'output_dim': 5,          # Number of case prediction classes
        'num_locations': 20,      # Number of geographic locations
        'num_gcn_layers': 3,      # Number of GCN layers
        'num_lstm_layers': 2,     # Number of LSTM layers
        'dropout': 0.1            # Dropout rate
    }
}
```

trainer = DeepEVDTrainer(config, use_wandb=True)

# Train model
history = trainer.train(train_loader, val_loader)

# Make predictions
predictions = model(node_features, adjacency_matrix)
```

### Training

```python
# Basic training script
python train.py --config config.yaml --epochs 100 --batch_size 32

# Training with custom parameters
python train.py \
    --config config.yaml \
    --epochs 200 \
    --batch_size 64 \
    --lr 0.001 \
    --dropout 0.2 \
    --patience 30
```

### Inference

```python
# Load trained model
model = DeepEVDModel.load_from_checkpoint('path/to/checkpoint.pth')
model.eval()

# Make predictions
with torch.no_grad():
    predictions = model(node_features, adjacency_matrix)
    
# Extract specific predictions
case_predictions = predictions['case_predictions']
r0_estimates = predictions['r0_estimates']
risk_maps = predictions['risk_maps']
```


adjacency_matrix = torch.tensor([...])

targets = {
    'case_targets': torch.tensor([...]),    # Case count categories
    'r0_targets': torch.tensor([...]),      # Reproduction rate values
    'risk_targets': torch.tensor([...])     # Risk level indicators
}
```

### Data Preprocessing

```python
from utils.preprocessing import EpidemicDataPreprocessor

preprocessor = EpidemicDataPreprocessor(config)
train_data = preprocessor.preprocess(raw_data, mode='train')
val_data = preprocessor.preprocess(raw_data, mode='val')
```

```yaml
model:
  input_dim: 8
  embedding_dim: 32
  spatial_dim: 64
  temporal_dim: 128
  output_dim: 5
  num_locations: 20
  num_gcn_layers: 3
  num_lstm_layers: 2
  dropout: 0.1

training:
  epochs: 100
  batch_size: 32
  validation_split: 0.2

optimizer:
  type: adamw
  lr: 0.001
  weight_decay: 0.0001

scheduler:
  type: cosine
  min_lr: 0.000001

loss_weights:
  case: 1.0
  r0: 0.5
  risk: 0.3

patience: 20
save_dir: ./checkpoints
l1_lambda: 0.00001
log_level: INFO
```


### Model Performance Metrics

- Classification Accuracy: For case category predictions
- AUC-ROC: For risk mapping performance
- Spatial Correlation: Spatial prediction consistency
- Temporal Stability: Prediction stability over time

## Results

## Performance on EVD Dataset

| Country | Time Horizon | MAE | RMSE | MAPE | R₀ MAE |
|---------|--------------|-----|------|------|--------|
| Sierra Leone | 15min | 3.25 | 6.51 | 10.45% | 0.08 |
| Sierra Leone | 30min | 3.58 | 7.30 | 18.45% | 0.12 |
| Sierra Leone | 60min | 5.49 | 7.21 | 10.89% | 0.15 |
| Guinea | 15min | 3.31 | 6.52 | 20.51% | 0.09 |
| Guinea | 30min | 4.82 | 6.30 | 15.64% | 0.13 |
| Guinea | 60min | 5.51 | 6.46 | 15.94% | 0.16 |
| Liberia | 15min | 3.32 | 6.35 | 17.56% | 0.08 |
| Liberia | 30min | 3.68 | 7.29 | 18.78% | 0.11 |
| Liberia | 60min | 5.54 | 7.30 | 20.45% | 0.17 |

### Visualization

```python
# Plot training curves
trainer.plot_training_curves('./results/training_curves.png')

# Generate prediction plots
from utils.visualization import plot_predictions
plot_predictions(model, test_data, save_path='./results/predictions.png')

# Create risk maps
from utils.visualization import create_risk_maps
create_risk_maps(predictions['risk_maps'], locations, save_path='./results/risk_maps.png')
```

## File Structure

```
deepevd/
├── model/
│   ├── __init__.py
│   ├── models.py              # Core model architecture
│   ├── layers.py              # Custom neural network layers
│   └── attention.py           # Attention mechanisms
├── training/
│   ├── __init__.py
│   ├── trainer.py             # Training pipeline
│   └── losses.py              # Custom loss functions
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py       # Data preprocessing utilities
│   ├── visualization.py       # Plotting and visualization tools
│   ├── metrics.py             # Evaluation metrics
│   └── data_loader.py         # Data loading utilities
├── config/
│   ├── default_config.yaml    # Default configuration
│   └── experiment_configs/    # Experimental configurations
├── scripts/
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   └── predict.py            # Prediction script
├── tests/
│   ├── test_model.py         # Model unit tests
│   ├── test_training.py      # Training pipeline tests
│   └── test_utils.py         # Utility function tests
├── examples/
│   ├── basic_usage.py        # Basic usage examples
│   └── advanced_examples.py  # Advanced usage examples
├── docs/
│   ├── API.md                # API documentation
│   └── tutorials/            # Tutorial notebooks
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
├── README.md                 # This file
└── LICENSE                   # License file
```

## Contributing

We welcome contributions to DeepEVD! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Write comprehensive unit tests
- Update documentation for new features
- Ensure compatibility with PyTorch 1.9+

## Citation

If you use DeepEVD in your research, please cite our paper:

```bibtex
@article{deepevd2024,
  title={DeepEVD: Deep Learning for Spatio-Temporal Epidemiological Forecasting},
  author={Author Name and Co-authors},
  journal={Journal Name},
  year={2025},
  volume={XX},
  pages={XXX-XXX},
  doi={10.xxxx/xxxxx}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- World Health Organization (WHO) for providing epidemiological data
- International Organisation for Migration (IOM-UN Migration Agency) for providing Flow Monitoring Points (FMPs) data
- Research collaborators from Guinea, Sierra Leone, and Liberia
- Open-source community for foundational tools and libraries

##Contact

- Author: [Abdul Joseph Fofanah]
- Email: [a.fofanah@griffith.edu.au]
- Institution: Griffith University, School of ICT
- Project Homepage: https://github.com/afofanah/DeepEVDMob 

Disclaimer: This software is for research purposes only. For actual epidemiological decision-making, please consult with qualified public health professionals and validate results with domain experts.
