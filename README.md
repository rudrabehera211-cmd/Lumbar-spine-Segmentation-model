# Lumbar Spine MRI Segmentation System

State-of-the-art deep learning system for automatic segmentation of lumbar vertebrae and intervertebral discs from MRI scans.

## Architecture

Supports 7 model architectures with automatic benchmarking:

| Model | Type | Params |
|-------|------|--------|
| U-Net | Baseline CNN | 30M+ |
| Attention U-Net | CNN + Attention Gates | 34M+ |
| UNet++ | Nested Skip Connections | 36M+ |
| UNETR | Transformer Encoder | 100M+ |
| Swin UNETR | Swin Transformer | 62M+ |
| nnU-Net | Self-configuring U-Net | 30M+ |
| ATM-Net (ours) | Cross-Attention + Multi-Scale | 35M+ |

## Installation

```bash
# Clone the repository
cd spine-mri-segmentation

# Create virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Structure

Place .mha files in `data/raw/` with format:
```
data/raw/
├── patient001_image.mha
├── patient001_mask.mha
├── patient002_image.mha
├── patient002_mask.mha
...
```

## Usage

### Quick Start (Full Pipeline)

```bash
python scripts/run_pipeline.py --mode full --data-dir ./data/raw
```

### Data Validation Only

```bash
python scripts/run_pipeline.py --mode validate --data-dir ./data/raw
```

### Dataset Analysis

```bash
python scripts/run_pipeline.py --mode analyze --data-dir ./data/raw
```

### Train a Model

```bash
python scripts/run_pipeline.py --mode train --data-dir ./data/raw --model-name atm_net
```

### Benchmark All Models

```bash
python scripts/run_pipeline.py --mode benchmark --data-dir ./data/raw
```

### Hyperparameter Optimization

```bash
python scripts/run_pipeline.py --mode hyperopt --data-dir ./data/raw
```

### Evaluate

```bash
python scripts/run_pipeline.py --mode evaluate --data-dir ./data/raw --checkpoint ./checkpoints/best.pt
```

### Inference

```bash
# Single file
python scripts/run_pipeline.py --mode infer --data-dir ./path/to/image.mha --checkpoint ./checkpoints/best.pt

# Folder
python scripts/run_pipeline.py --mode infer --data-dir ./input_folder --checkpoint ./checkpoints/best.pt
```

### Run Tests

```bash
python -m pytest tests/
# Or individually:
python tests/test_data_pipeline.py
python tests/test_models.py
python tests/test_losses.py
python tests/test_inference.py
```

## Deployment

### FastAPI

```bash
python -m deployment.fastapi_app
# API at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Gradio

```bash
python -m deployment.gradio_app
# UI at http://localhost:7860
```

### Streamlit

```bash
streamlit run deployment/streamlit_app.py
# UI at http://localhost:8501
```

## Label Map

| Label | Structure |
|-------|-----------|
| 0 | Background |
| 1 | L1 Vertebra |
| 2 | L2 Vertebra |
| 3 | L3 Vertebra |
| 4 | L4 Vertebra |
| 5 | L5 Vertebra |
| 6 | Sacrum |
| 7 | Disc L1-L2 |
| 8 | Disc L2-L3 |
| 9 | Disc L3-L4 |
| 10 | Disc L4-L5 |
| 11 | Disc L5-S1 |

## Configuration

See `configs/default_config.yaml` for all configurable parameters.

Key settings:
- `model.name`: Architecture choice
- `training.loss.name`: Loss function
- `preprocessing.target_size`: Volume resizing
- `augmentation.*`: Augmentation parameters
- `training.mixed_precision`: Enable AMP

## Output Structure

```
reports/
├── run_YYYYMMDD_HHMMSS/
│   ├── checkpoints/
│   │   ├── best.pt
│   │   └── epoch_*.pt
│   ├── visualizations/
│   │   └── *.png
│   ├── confusion_matrix.png
│   ├── training_history.json
│   ├── test_metrics.json
│   └── error_analysis.json
├── data_validation_report.json
├── data_split.json
└── class_distribution.json
```

## Features

- [x] 2D and 3D training support
- [x] Mixed precision training (AMP)
- [x] Gradient accumulation and clipping
- [x] Warmup + Cosine LR scheduler
- [x] Early stopping
- [x] Checkpointing with resume
- [x] 7 loss functions with auto-selection
- [x] 7 model architectures with auto-benchmark
- [x] Deep supervision
- [x] Multi-scale context fusion
- [x] Cross-attention fusion
- [x] Test-time augmentation (TTA)
- [x] Data integrity validation
- [x] Class distribution analysis
- [x] Error analysis with recommendations
- [x] Per-class metrics (Dice, IoU, Precision, Recall, HD95, ASD)
- [x] Confusion matrix generation
- [x] Learning curves
- [x] Hyperparameter optimization (Optuna)
- [x] FastAPI, Gradio, Streamlit deployment
- [x] .mha and PNG output formats
