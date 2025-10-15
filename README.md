# DARSA: Domain Adaptation Reaction-Informed for Synthetic Accessibility Assessment

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A domain-adaptive deep learning framework for predicting synthetic accessibility of drug-like molecules across diverse chemical spaces.

## 🎯 Overview

DARSA addresses the critical challenge of **domain shift** in synthetic accessibility prediction by combining:
- **Domain Adversarial Training** with gradient reversal layers
- **Graph Attention Networks** for molecular representation learning
- **Reaction Database Knowledge** from 6M+ reactions (USPTO + ORD)

### Key Features

✅ **Cross-Domain Robustness**: Maintains AUROC >0.93 across diverse chemical domains  
✅ **High Precision**: >94% precision reduces false positive predictions by 25%  
✅ **Computational Efficiency**: 23.4 ms per molecule for high-throughput screening  
✅ **Expert Validation**: r=0.970 correlation with medicinal chemist evaluations  
✅ **Interpretable Predictions**: Chemical logic validation through synthesis routes

---

## 📊 Performance Highlights

| Test Set | Molecules | AUROC | Precision | Domain Challenge |
|----------|-----------|-------|-----------|------------------|
| TS1 | 7,162 | 1.000 | 0.981 | Moderate (61.1% overlap) |
| TS2 | 30,348 | 0.963 | 0.943 | Severe shift (82.6% overlap) |
| TS3 | 1,800 | 0.933 | 0.958 | Extreme (85.4% overlap) |

**Comparison**: While SCScore drops to AUROC 0.442 on cross-domain tests, DARSA maintains robust performance.

---

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/DARSA.git
cd DARSA

# Create conda environment
conda create -n darsa python=3.9
conda activate darsa

# Install dependencies
pip install -r requirements.txt
```

### Requirements
```
torch>=1.12.1
torch-geometric>=2.1.0
rdkit>=2023.03.1
scikit-learn>=1.1.3
numpy>=1.23.0
pandas>=1.5.0
```

### Quick Prediction
```python
from DARSA_score import predict_sa

# Predict synthetic accessibility for a SMILES string
smiles = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"  # Ibuprofen
score = predict_sa(smiles)
print(f"Synthetic Accessibility Score: {score:.2f}")
# Output: Score range 1-10 (1=easy, 10=difficult)
```

---

## 📁 Repository Structure
```
DARSA/
├── best_model.pth              # Pre-trained DARSA model weights
├── DARSA_model.py              # Main model architecture (GAT + DANN)
├── DARSA_score.py              # Prediction interface
├── train.py                    # Training script
├── test_model.py               # Evaluation script
├── data_preprocess.py          # Data preprocessing pipeline
├── ReactionScores.py           # Reaction fragment scoring module
├── Reaction_USPTO.pkl.gz       # USPTO reaction database (3.7M reactions)
├── Reaction_ORD.pkl.gz         # Open Reaction Database (2.3M reactions)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🔬 Usage Examples

### 1. Batch Prediction
```python
from DARSA_score import batch_predict

# List of SMILES strings
smiles_list = [
    "CC(C)Cc1ccc(cc1)C(C)C(O)=O",  # Ibuprofen
    "CC(=O)Oc1ccccc1C(=O)O",        # Aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # Caffeine
]

scores = batch_predict(smiles_list)
for smiles, score in zip(smiles_list, scores):
    print(f"{smiles}: {score:.2f}")
```

### 2. Training from Scratch
```python
# Prepare your dataset (SMILES + labels)
# Format: CSV with columns ['smiles', 'label']
# label: 0 = Easy to Synthesize (ES), 1 = Hard to Synthesize (HS)

python train.py \
    --train_data path/to/train.csv \
    --val_data path/to/val.csv \
    --epochs 100 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --save_dir ./checkpoints
```

### 3. Model Evaluation
```python
python test_model.py \
    --model_path best_model.pth \
    --test_data path/to/test.csv \
    --output results.csv
```

### 4. Data Preprocessing
```python
from data_preprocess import preprocess_dataset

# Convert raw SMILES to molecular graphs
preprocess_dataset(
    input_file='raw_data.csv',
    output_file='processed_data.pkl',
    augment=True  # Generate 2-4 SMILES variants per molecule
)
```

---

## 🏗️ Model Architecture

DARSA integrates three complementary components:

### 1. Domain-Adaptive Graph Attention Networks
- **8 attention heads** with 128-dimensional hidden layers
- **3 GAT layers** with residual connections
- **Node features**: 12D (atomic properties, hybridization, chirality)
- **Edge features**: 5D (bond type, conjugation, stereochemistry)

### 2. Reaction Fragment Scoring
- **6M+ reactions** from USPTO and ORD databases
- **Morgan fingerprints** (radius=2, 2048 bits, chirality-aware)
- **Multi-component scoring**: fragments + complexity + density + patterns

3. Domain Adversarial Training (DANN)
- **Gradient Reversal Layer** (α=1.0) for domain-invariant learning
- **Domain classifier convergence**: 85% → 52% (domain confusion)
- **Ensemble weights**: W_RF=0.731, W_GNN=0.134, W_CK=0.135

---

📈 Reproducing Paper Results

 Download Datasets
```bash
# External test sets used in the paper
wget https://example.com/TS1_SYBA_benchmark.csv
wget https://example.com/TS2_RAscore_benchmark.csv
wget https://example.com/TS3_challenging_pairs.csv
```

Run Evaluation
```bash
# Test Set 1 (SYBA Benchmark)
python test_model.py --test_data TS1_SYBA_benchmark.csv

# Test Set 2 (RAscore Benchmark)
python test_model.py --test_data TS2_RAscore_benchmark.csv

# Test Set 3 (Challenging Pairs)
python test_model.py --test_data TS3_challenging_pairs.csv
```

---

 🎓 Citation

If you use DARSA in your research, please cite:
```bibtex
@article{aljanabi2025darsa,
  title={DARSA: Domain Adaptation Reaction-Informed for Synthetic Accessibility Assessment},
  author={Aljanabi, Qahtan Adnan and Deng, Lei},
  journal={Journal of Chemical Information and Modeling},
  year={2025},
  publisher={American Chemical Society}
}
```



🔍 Key Advantages

1. Cross-Domain Generalization
Traditional methods fail catastrophically on new chemical domains (SCScore: 0.717→0.442 AUROC). DARSA maintains >0.93 AUROC across all domains through adversarial domain adaptation.

2. High Precision for Pharmaceutical Applications
94% precision means pharmaceutical teams can trust positive predictions, reducing synthesis failures by ~25% and saving weeks of chemistry time.

3. Expert-Level Accuracy
Correlation r=0.970 with expert medicinal chemists (n=7, 14.2±3.7 years experience) on 40-molecule Ertl benchmark.

4. Chemical Interpretability
Unlike black-box models, DARSA's predictions can be traced to specific molecular features and reaction patterns.

---

 ⚠️ Limitations

DARSA's performance degrades for:
- **Novel scaffolds**: Tanimoto similarity <0.3 with training set
- **High stereochemical complexity**: Molecules with >6 stereocenters
- **Large macrocycles**: Rings >12 atoms (limited training data)
- **Organometallic compounds**: Underrepresented in reaction databases

---
 🛠️ Advanced Usage
 Custom Training with Your Data
```python
from DARSA_model import DARSA
from torch.utils.data import DataLoader

# Initialize model
model = DARSA(
    node_features=12,
    edge_features=5,
    hidden_dim=128,
    num_heads=8,
    num_layers=3,
    dropout=0.2
)

# Custom training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(100):
    for batch in train_loader:
        loss = model.train_step(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

 Fine-tuning for Specific Domains
```python
# Load pre-trained model
model = DARSA.load_pretrained('best_model.pth')

# Fine-tune on your domain-specific data
model.finetune(
    domain_data='your_domain.csv',
    epochs=20,
    freeze_encoder=False  # Set True to freeze GAT layers
)
```

---




This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.


---

🙏 Acknowledgments

- **Reaction Databases**: USPTO (Lowe, 2012) and Open Reaction Database (Kearnes et al., 2021)
- **Benchmark Datasets**: SYBA (Voršilák et al., 2020), RAscore (Thakkar et al., 2021)
- **Expert Validation**: Seven medicinal chemists for evaluation support
- **Computing Resources**: CSU High-Performance Computing Center

---

## 📧 Contact

For questions, issues, or collaborations:
- **Email**: 214708021@csu.edu.cn
- **GitHub Issues**: [Create an issue](https://github.com/yourusername/DARSA/issues)
- **Phone**: +86 150-7485-1532

---

📚 References

1. Ertl, P. & Schuffenhauer, A. (2009). Estimation of synthetic accessibility score. *J. Cheminform.*, 1:8.
2. Coley, C.W. et al. (2018). SCScore: Synthetic Complexity Learned from a Reaction Corpus. *J. Chem. Inf. Model.*, 58(2):252-261.
3. Kearnes, S.M. et al. (2021). The Open Reaction Database. *J. Am. Chem. Soc.*, 143(45):18820-18826.
4. Lowe, D.M. (2012). Extraction of Chemical Structures and Reactions from the Literature. PhD Thesis, University of Cambridge.

---

 📈 Project Status

- [x] Initial release with pre-trained model
- [x] Paper accepted for publication
- [ ] Web interface development
- [ ] Docker image release
- [ ] API endpoint deployment
- [ ] Integration with popular cheminformatics platforms



⭐ Star this repository if you find it useful!**

Last updated: January 2025

