# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Thesis project on **Aspect-Based Sentiment Analysis (ABSA)** for Indonesian tourism reviews. Implements a hierarchical multi-task learning architecture (Wang et al., 2021) adapted for IndoBERT, comparing it against a single-task IndoBERT+CRF baseline.

**Base model**: `indobenchmark/indobert-large-p2` (1024-dim, 24 layers, ~330M params)
**Hardware constraint**: NVIDIA RTX 4060 Laptop (8GB VRAM) — batch size 4 with gradient accumulation ×4 = effective batch 16.

## Architecture

### Multi-Task Model (Notebooks 03, 04, 06, 08, 09)

Four hierarchical tasks sharing an IndoBERT encoder:
1. **ATE** (Aspect Term Extraction): BIEOS sequence labeling with word-level prior embedding
2. **SLD** (Sentiment Lexicon Detection): O/POS/NEG token classification
3. **ASD** (Aspect Sentiment Detection): Cross-attention from ATE→SLD with relative position encoding → O/POS/NEG/NEU
4. **Joint CRF**: Concatenated ATE+ASD projections → 13-class BIEOS-sentiment CRF

**Loss**: `L = λ1*(L_ATE + L_SLD) + λ2*L_ASD + L_CRF` (default λ1=λ2=0.3)

**Two-phase training**: Phase 1 pre-trains SLD (3 epochs) → sentiment connection copies SLD→ASD weights → Phase 2 full training (15+ epochs).

### Single-Task Baseline (Notebook 07)

`IndoBERT → Dropout → Linear(1024→256) → GELU → Dropout → Linear(256→13) → CRF`

No auxiliary tasks, prior embedding, or cross-attention. Currently the best performer (F1: 0.7551 vs multi-task 0.7410).

## Data Pipeline

```
Raw ABSA reviews (2,451 samples, BIEOS labels with POS/NEG/NEU sentiment)
  → NB 01: Slang normalization, BIEOS conversion, filtering (<3 tokens removed)
  → NB 02: IndoBERT subword tokenization, label alignment (first subword keeps label, rest=-100), prior embedding
  → NB 05: Augmentation on train split only (NEG×3, NEU×2, O-token perturbation → 4,290 train samples)
```

Train/val split: 85/15 (seed=42), augmentation applied only to train to prevent leakage.

## Label Scheme

```python
ate_labels   = ['O', 'B', 'I', 'E', 'S']               # 5 classes
sld_labels   = ['O', 'POS', 'NEG']                      # 3 classes
asd_labels   = ['O', 'POS', 'NEG', 'NEU']               # 4 classes
bieos_labels = ['O', 'B-POS', 'I-POS', 'E-POS', 'S-POS',
                'B-NEG', 'I-NEG', 'E-NEG', 'S-NEG',
                'B-NEU', 'I-NEU', 'E-NEU', 'S-NEU']     # 13 classes
IGNORE_INDEX = -100  # padding tokens excluded from loss
```

## Key Files

| Path | Description |
|------|-------------|
| `data/raw/ABSA_dataset.txt` | Original annotated reviews |
| `data/raw/lexicon_*.csv` | Indonesian sentiment lexicon (POS/NEG word lists) |
| `data/processed/train_data_bieos.json` | Preprocessed 2,451 samples |
| `data/processed/dataset_indobert_bieos.pt` | Tokenized tensors ready for training |
| `data/processed/train_data_bieos_augmented_train.json` | Augmented train split (4,290) |
| `models/best_model*.pt` | Saved model weights (~1.3GB each) |

## Code Conventions

- **Path setup in notebooks**: `BASE_DIR = os.path.dirname(os.getcwd())` (notebooks run from `notebooks/` dir)
- **Reproducibility**: `torch.manual_seed(SEED)` + `torch.cuda.manual_seed_all(SEED)` + `random.seed(SEED)` + `np.random.seed(SEED)`
- **Evaluation**: Entity-level F1 via `seqeval`, ignoring padding index (-100)
- **Early stopping**: Track best val F1, restore best weights, patience typically 5 epochs

## Dependencies

PyTorch, Hugging Face Transformers, torchcrf, seqeval, pandas, numpy, matplotlib, scikit-learn. Max token length: 128.
