# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Thesis project on **Aspect-Based Sentiment Analysis (ABSA)** for Indonesian tourism reviews. Implements a hierarchical multi-task learning architecture (Wang et al., 2021) adapted for IndoBERT, comparing it against a single-task IndoBERT+CRF baseline.

**Base model**: `indobenchmark/indobert-large-p2` (1024-dim, 24 layers, ~330M params)
**Hardware constraint**: NVIDIA RTX 4060 Laptop (8GB VRAM) — batch size 4 with gradient accumulation ×4 = effective batch 16.

## Environment

Conda environment: `ta_nlp` (`conda activate ta_nlp`). Key deps: PyTorch 2.5.1 (CUDA 12.1), Transformers, torchcrf, seqeval, pandas, numpy, matplotlib, scikit-learn. Max token length: 128.

## Notebook Guide

All code lives in Jupyter notebooks (no shared Python modules, `src/` is empty). Run from `notebooks/` directory.

| NB | Filename | Purpose |
|----|----------|---------|
| 00 | `00_Setup_Check` | Environment/CUDA setup check |
| 01 | `01_Preprocessing` | Slang normalization, BIEOS conversion, filtering |
| 02 | `02_Tokenization` | IndoBERT tokenization, label alignment, prior embedding |
| 03 | `03_MultiTask_Training` | **Multi-task training** (Phase 1 SLD pre-train + Phase 2 full) → F1=0.7410 |
| 04 | `04_Hyperparameter_Search` | Grid search (λ1, λ2, dropout, LR) |
| 05 | `05_Data_Augmentation` | NEG×3, NEU×2, O-token perturbation → 4,290 train |
| 06 | `06_MultiTask_Augmented` | MT + augmented data + freeze 12 layers → F1=0.7341 |
| 07 | `07_SingleTask_Baseline` | **Single-task baseline** (IndoBERT+CRF only) → F1=0.7551 |
| 08 | `08_Lambda_Phase_Tuning` | 4 skenario λ dan phase config |
| 09 | `09_MultiSeed_Robustness` | Seeds 42/123/456/789/999 |
| 10 | `10_Extended_Training_50ep` | 4 runs × 50 epochs (MT/ST × Orig/Aug) → **best F1=0.7600** |
| 11 | `11_Ablation_Study` | Three-level: Softmax vs CRF vs MTL+CRF |
| 12 | `12_Base_Model_Comparison` | indobert-large-p2 vs alternatives |
| 13 | `13_MTL_Architecture_Variants` | Label smoothing, task dropout, task-alternating, partial layer sharing |
| 14 | `14_Evaluation_ErrorAnalysis` | Confusion matrices, error analysis, robustness |
| 15 | `15_Visualization_Interpretability` | Learning curves, attention weights, t-SNE, heatmaps |

## Architecture

### Multi-Task Model (NB 03, 04, 06, 08–13)

```
IndoBERT (shared encoder, all 24 layers trainable)
        │
        h (1024-dim)
       / \
  [h;prior] h
      │      │
    h_ae   h_sl  ← ATE & SLD projections (1024→256 + GELU + Dropout)
     │       │
  ATE cls  SLD cls  ← CrossEntropyLoss (λ1 weight)
     │       │
     └──→ CrossAttention(ATE→SLD) + relative position encoding
              │
            h_sd
              │
          ASD cls  ← CrossEntropyLoss (λ2 weight)
              │
       [h_ae ; h_sd]  ← Concatenate
              │
        Linear(512→256) + GELU + Dropout → Linear(256→13)
              │
          CRF(13 classes)  ← NLL loss
```

**Loss** (Paper Eq. 10): `L = λ1*(L_ATE + L_SLD) + λ2*L_ASD + L_CRF` (default λ1=λ2=0.3)

**Two-phase training**: Phase 1 pre-trains SLD (3 epochs) → sentiment connection copies SLD→ASD weights → Phase 2 full training (15+ epochs).

### Single-Task Baseline (NB 07, 10, 11)

`IndoBERT → Dropout(0.1) → Linear(1024→256) → GELU → Dropout(0.1) → Linear(256→13) → CRF`

No auxiliary tasks, prior embedding, or cross-attention.

## Default Hyperparameters

```python
SEED = 42
LR_BERT = 2e-5          # Encoder learning rate
LR_HEAD = 1e-4          # Task head learning rate
DROPOUT = 0.1
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
WARMUP_RATIO = 0.1      # Fraction of total steps
LAMBDA1 = 0.3           # ATE + SLD loss weight
LAMBDA2 = 0.3           # ASD loss weight
PROJ_DIM = 256           # Projection dimension
MAX_REL_POS = 20         # Cross-attention relative position encoding
BATCH_SIZE = 4           # Per-device (×4 grad accum = 16 effective)
MAX_LEN = 128            # Token sequence length
PHASE1_EPOCHS = 3        # SLD pre-training
NUM_EPOCHS = 15          # Full training (50 in NB 10+)
PATIENCE = 5             # Early stopping (10 in NB 10+)
```

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
# NOTE: bieos_labels are alphabetically sorted in actual code:
bieos_labels = ['O', 'B-NEG', 'B-NEU', 'B-POS', 'E-NEG', 'E-NEU', 'E-POS',
                'I-NEG', 'I-NEU', 'I-POS', 'S-NEG', 'S-NEU', 'S-POS']  # 13 classes
IGNORE_INDEX = -100  # padding tokens excluded from loss
```

## Key Files

| Path | Description |
|------|-------------|
| `data/raw/ABSA_all_2500data_train.txt` | Original 2,451 annotated reviews |
| `data/raw/indonesian_sentiment_lexicon_*.tsv` | POS (2,288) and NEG (5,025) word lists |
| `data/raw/colloquial-indonesian-lexicon.csv` | Indonesian slang dictionary |
| `data/processed/train_data_bieos.json` | Preprocessed 2,451 samples |
| `data/processed/dataset_indobert_bieos.pt` | Tokenized tensors ready for training |
| `data/processed/train_data_bieos_augmented_train.json` | Augmented train split (4,290) |
| `data/processed/train_data_bieos_val.json` | Validation split (368 samples) |
| `models/best_model.pt` | Multi-task best weights |
| `models/best_model_singletask.pt` | Single-task best weights |

## Key Results

| Model | Dataset | F1 | NB |
|-------|---------|-----|-----|
| Multi-task | Original | 0.7410 | 03 |
| Single-task | Original | 0.7551 | 07 |
| MT multi-seed | Original | 0.7319 ± 0.0067 | 09 |
| ST multi-seed | Original | 0.7297 ± 0.0166 | 09 |
| MT extended (50 ep) | Original | 0.7578 | 10 |
| MT extended (50 ep) | Augmented | 0.7442 | 10 |
| ST extended (50 ep) | Original | 0.7558 | 10 |
| **ST extended (50 ep)** | **Augmented** | **0.7600** | **10** |

Prior research baseline (EMC-GCN): 0.6761.

### Three-Level Ablation (NB 11)

| Level | Architecture | Orig F1 | Aug F1 |
|-------|-------------|---------|--------|
| L1 | IndoBERT + Softmax | 0.7466 | 0.7593 |
| L2 | IndoBERT + CRF | 0.7558 | 0.7600 |
| L3 | Hierarchical MTL + CRF | 0.7578 | 0.7442 |

CRF adds +0.0092 on original data. MTL adds +0.0021 on original but **hurts** on augmented (−0.0158).

### MTL Variants (NB 13) — none improved over MT base (0.7578)

| Variant | F1 | Delta |
|---------|-----|-------|
| Label smoothing=0.1 | 0.7421 | −0.0157 |
| Task-specific dropout=0.2 | 0.7493 | −0.0085 |
| Task-alternating training | 0.7392 | −0.0187 |
| Partial layer sharing (top 3 BERT layers split) | 0.7487 | −0.0091 |

## Code Conventions

- **Path setup**: `BASE_DIR = os.path.dirname(os.getcwd())` (notebooks run from `notebooks/` dir)
- **Reproducibility**: `torch.manual_seed(SEED)` + `torch.cuda.manual_seed_all(SEED)` + `random.seed(SEED)` + `np.random.seed(SEED)` with SEED=42
- **Evaluation**: Entity-level F1 via `seqeval`, ignoring padding index (-100)
- **Early stopping**: Track best val F1, restore best weights, patience=5 (NB 07) or patience=10 (NB 10+)
- **GPU memory**: Between runs, use `del model; gc.collect(); torch.cuda.empty_cache()` to reclaim VRAM
- **No shared utility modules**: All model definitions, training loops, and evaluation code are self-contained within each notebook
- **Checkpointing**: Save `*_intermediate.pt` during training and `checkpoint_*.pt` / `best_model_*.pt` at completion
