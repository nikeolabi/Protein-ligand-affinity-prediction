# Protein-Ligand Affinity Prediction using Deep Learning

**Author**: Nike Olabiyi  
**Created**: January 2026  

## Project Overview

This project implements deep learning models for predicting protein-ligand binding affinity, a critical task in computational drug discovery. The project explores two main approaches:

1. **DeepDTA-based CNN Architecture** (Part 1) - Character-level convolutional neural networks
2. **ESM-2 + Morgan Fingerprints** (Part 2) - Pre-trained protein language models with chemical fingerprints

## Repository Structure

```
protein-ligand_affinity/
├── Project_Protein_Ligand_Affinity Part 1.ipynb  # DeepDTA implementation
├── Project_Protein_Ligand_Affinity Part 2.ipynb  # ESM-2 + Morgan approach
├── README.md                                            # This file
├── data/                                                # Dataset files (not included)
│   ├── train.csv
│   ├── test_task1.csv
│   ├── test_task2_transfer.csv
│   └── test_task3_screening.csv
├── cached_embeddings/                                   # ESM-2 protein embeddings cache
│   ├── protein_embeddings.pkl                           # Cached ESM-2 embeddings
│   └── embedding_metadata.json                          # Embedding configuration info
└── saved_models/                                        # Saved models (generated)
    ├── deepdta_model_1.keras                             # Model 1
    ├── deepdta_model_2.keras                            # Model 2  
    ├── deepdta_model_3.keras                            # Model 3
    ├── deepdta_model_cnn_lstm_001.keras                 # Model 4
    ├── deepdta_weights.weights.h5
    ├── ...
    └── checkpoints/                                    # Model checkpoints (used for Model 4)

```

## Approaches

### Part 1: DeepDTA-based CNN Architecture

This approach implements a character-level CNN model inspired by the DeepDTA paper:
https://pmc.ncbi.nlm.nih.gov/articles/PMC6129291/

**Architecture Features:**
- **Protein Branch**: 3 Conv1D layers (32→64→96 filters, kernel size 8) + GlobalMaxPooling
- **Ligand Branch**: 3 Conv1D layers (32→64→96 filters, kernel size 4) + GlobalMaxPooling
- **Combined Features**: Concatenation → Dense layers (1024→1024→512→1)
- **Regularization**: Dropout (0.1-0.3), Early Stopping
- **Input Representations**: Character-level encoding for both proteins and SMILES

**Key Implementation Details:**
- Maximum protein sequence length: 1000 (99th percentile)
- Maximum SMILES length: 100
- Vocabulary-based embedding layers
- Adam optimizer with MSE loss
- MAE monitoring metric

**Model Variants Tested:**
1. **Model 1**: Original DeepDTA architecture with 0.1 dropout
2. **Model 2**: Adjusted version with increased dropout (0.3) and smaller dense layers to reduce overfitting
3. **Model 3**: Enhanced version with L2 regularization, improved early stopping, and learning rate scheduling
4. **Model 4**: Hybrid CNN-LSTM architecture combining convolutional and recurrent layers

### Model 3: Enhanced DeepDTA with Regularization

Builds upon Model 2 with additional regularization techniques:

**Enhanced Features:**
- **L2 Regularization**: Light regularization (0.0001) applied to all dense layers
- **Improved Early Stopping**: Patience of 8 epochs, monitoring val_mae with min_delta=0.001
- **Learning Rate Scheduling**: ReduceLROnPlateau with factor=0.7, patience=5
- **Same Architecture**: 512→256→128 dense layers with 0.3 dropout

**Regularization Strategy:**
- Kernel regularization on dense layers to prevent overfitting
- Adaptive learning rate reduction for better convergence
- More aggressive early stopping for optimal model selection

### Model 4: Hybrid CNN-LSTM Architecture

Advanced architecture combining convolutional and recurrent neural networks:

**Architecture Features:**
- **Protein Branch**: Conv1D layers (64→96 filters, kernel size 8) → Bidirectional LSTM (64 units)
- **Ligand Branch**: Conv1D layers (64→96 filters, kernel size 4) → Bidirectional LSTM (64 units)
- **Sequential Processing**: CNN extracts local patterns, LSTM captures long-range dependencies
- **Dense Layers**: 256→128 nodes with 0.3 dropout

**Technical Implementation:**
- **Custom MaskedConv1D**: Preserves padding masks for LSTM compatibility
- **Bidirectional LSTM**: Processes sequences in both directions
- **Enhanced Checkpointing**: Automatic model saving every 10 epochs
- **Standard Learning Rate**: Adam optimizer with default 0.001 learning rate


### Part 2: ESM-2 + Morgan Fingerprints

This approach leverages state-of-the-art pre-trained models:

**Architecture Features:**
- **Protein Representation**: ESM-2 (esm2_t12_35M_UR50D) pre-trained language model
- **Ligand Representation**: Morgan fingerprints (RDKit)
- **Model**: Dense neural network for regression
- **Frozen ESM-2**: Uses ESM-2 in inference mode with frozen parameters

**Key Implementation Details:**
- ESM-2 model size: 35M parameters
- Morgan fingerprint radius: 2, bit length: 2048
- Mean pooling for ESM-2 sequence representations
- TensorFlow/Keras backend with mixed precision support

**Computational Limitations:**
- ESM-2 processing proved more resource-intensive than anticipated
- Successfully processed only 6,552 out of 153,777 available protein sequences
- Final dataset: 6,552 training samples, 700 validation samples, 700 test samples


## Technical Requirements

### Dependencies
```python
# Core ML libraries
tensorflow >= 2.x
keras >= 3.x
torch
transformers

# Data processing
pandas
numpy
scikit-learn

# Molecular processing
rdkit

# Visualization
matplotlib
seaborn

# Utilities
pickle
```

### Hardware Requirements
- **GPU**: Recommended for training (CUDA-compatible)
- **RAM**: Minimum 16GB recommended
- **Storage**: ~5GB for datasets and models

## Dataset

The project uses protein-ligand binding affinity data with the following structure:
- **Training set**: 153,777 samples (60.0%)
- **Validation set**: 51,259 samples (20.0%)
- **Test set**: 51,260 samples (20.0%)
- **Total dataset**: 256,296 samples

**Specialized Benchmark Test Sets**: Independent evaluation datasets for specific tasks
  - **Task 1**: Standard benchmark test set (2,000 samples)
  - **Task 2**: Transfer learning evaluation set  
  - **Task 3**: Virtual screening benchmark set

*Note: These benchmark sets are separate from the main dataset and used for final model evaluation on specialized tasks.*

**Data Format:**
```csv
protein_sequence,   compound_smiles,   label
MKVLWAALLVTFLAGCQAKVEQAVETEPEPEL...,   CC(C)CC1=CC=C(C=C1)C(C)C(=O)O,    5.23
```

Where `label` represents pIC50 values (negative log of IC50 in molar units).

## Key Features

### Data Processing Pipeline
1. **Data Validation**: Check for missing values, duplicates, and data quality issues
2. **Sequence Length Analysis**: Statistical analysis of protein and SMILES lengths
3. **Duplicate Handling**: Averaging duplicate protein-ligand pairs
4. **Train-Validation-Test Split**: Proper data splitting with reproducible seeds

### Model Training Infrastructure
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Model Checkpointing**: Automatic saving of best models (only for Model 4)
- **Mixed Precision**: Automatically enabled for compatible GPUs (compute capability > 6)
- **Reproducibility**: Fixed random seeds for consistent results

### Evaluation Metrics
- **Mean Absolute Error (MAE)**: Primary evaluation metric
- **Mean Squared Error (MSE)**: Training loss function
- **R² Score**: Correlation analysis
- **Pearson Correlation**: Statistical correlation assessment (used only for Part 2)

## Performance Benchmarks

The models are evaluated against literature benchmarks:
- **DeepDTA Reference**: ~0.26 MSE when evaluated on non-similar datasets
- **Validation-Test Gap**: Monitor for overfitting assessment

## Usage

### Running Part 1 (DeepDTA CNN)
1. Open `Project_Protein_Ligand_Affinity Part 1.ipynb`
2. Configure data paths in the notebook
3. Execute cells sequentially for:
   - Data loading and preprocessing
   - Model definition and compilation
   - Training with monitoring
   - Evaluation on test sets

### Running Part 2 (ESM-2 + Morgan)
1. Open `Project_Protein_Ligand_Affinity Part 2.ipynb`
2. Install additional dependencies (RDKit, Transformers)
3. Execute cells for:
   - ESM-2 model loading
   - Feature extraction pipeline
   - Model training and evaluation

## Environment Setup

### Google Colab
Both notebooks are designed to run on Google Colab with GPU acceleration:
```python
# Mount Google Drive for data access
from google.colab import drive
drive.mount('/content/drive')
```

### Kaggle
Compatible with Kaggle notebook environment with automatic dependency detection.

### Local Setup
For local execution:
1. Install required dependencies
2. Update data paths to local directories
3. Configure GPU settings if available

## Model Interpretability

### Feature Analysis
- Protein sequence length distributions and their impact on predictions (truncation effects, optimal lengths)
- SMILES complexity metrics and binding affinity correlations (molecular weight, ring count, functional groups)
- Attention visualization for important sequence regions (ESM-2 only - active sites, binding domains)

### Performance Analysis
- Training curves for loss and metric monitoring
- Prediction scatter plots against experimental values
- Error distribution analysis across affinity ranges

## Future Improvements

1. **Architecture Enhancements**:
   - Attention mechanisms for better sequence modeling
   - Graph neural networks for molecular representations
   - Multi-task learning for related prediction tasks

2. **Data Augmentation**:
   - Protein structure information integration (3D coordinates, secondary structure, binding pockets)
   - Chemical space expansion techniques (SMILES enumeration, molecular conformations, stereoisomers)
   - Cross-dataset transfer learning (multi-dataset training, domain adaptation)

3. **Hyperparameter Optimization**:
   - Grid search for optimal learning rates and batch sizes
   - Bayesian optimization for regularization parameters
   - Automated architecture search (NAS) for model design

4. **Evaluation Robustness**:
   - Cross-validation strategies
   - External dataset validation
   - Uncertainty quantification

## License

The ESM-2 components are used under Meta's MIT License.  
Project code follows academic fair use principles.

## References

1. **DeepDTA**: Özlem Tastan, Arzucan Özgür. "DeepDTA: deep drug-target binding affinity prediction." *Bioinformatics*, 2018.
2. **ESM-2**: Lin et al. "Evolutionary-scale prediction of atomic level protein structure with a language model." *Science*, 2023.
3. **Morgan Fingerprints**: Rogers & Hahn. "Extended-connectivity fingerprints." *J. Chem. Inf. Model.*, 2010.
4. **Dataset**: Kaggle, "Structure-free protein–ligand affinity prediction." Available: https://www.kaggle.com/competitions/protein-compound-affinity/overview
