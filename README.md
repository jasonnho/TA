# IndoBERT-HMTL: Hierarchical Multi-Task Learning for Indonesian ABSA

Aspect-Based Sentiment Analysis (ABSA) pada ulasan desa wisata Indonesia menggunakan arsitektur **Hierarchical Multi-Task Learning** (Wang et al., 2021) yang diadaptasi untuk **IndoBERT**.

## Hasil Utama

| Model | Dataset | F1 |
|-------|---------|-----|
| EMC-GCN (prior research) | Original | 0.6761 |
| IndoBERT + Softmax | Original | 0.7466 |
| IndoBERT + CRF (Single-Task) | Original | 0.7558 |
| IndoBERT-HMTL (Multi-Task) | Original | 0.7578 |
| **IndoBERT + CRF + Augmented** | **Augmented** | **0.7600** |

Semua konfigurasi IndoBERT mengungguli baseline EMC-GCN (+7–8% F1).

## Arsitektur

### Multi-Task (Hierarchical)

```
IndoBERT (shared encoder, 24 layers)
        |
    [h ; prior_embedding]
       / \
    ATE    SLD          <- Parallel auxiliary tasks
     \    /
   Cross-Attention      <- Aspect-sentiment interaction
        |
      ASD               <- Aspect sentiment detection
        |
   [h_ae ; h_sd]
        |
      CRF (13-class)    <- Joint BIEOS-sentiment decoding
```

**Loss**: `L = λ1*(L_ATE + L_SLD) + λ2*L_ASD + L_CRF`

### Single-Task (Baseline)

```
IndoBERT → Linear → GELU → Linear → CRF (13-class)
```

## Dataset

- **2,451** ulasan wisata berbahasa Indonesia (format BIEOS + sentimen POS/NEG/NEU)
- Split: 85% train / 15% val (seed=42)
- Augmentasi: NEG×3, NEU×2, O-token perturbation → **4,290** train samples

## Setup

```bash
conda activate ta_nlp
```

Dependencies: PyTorch 2.5.1 (CUDA 12.1), Transformers, torchcrf, seqeval, pandas, numpy, matplotlib, scikit-learn.

Hardware: NVIDIA RTX 4060 Laptop (8GB VRAM), batch size 4 × 4 gradient accumulation = effective 16.

## Notebooks

Semua code ada di Jupyter notebooks. Jalankan dari direktori `notebooks/`.

| # | Notebook | Deskripsi |
|---|----------|-----------|
| 00 | Setup_Check | Cek environment & CUDA |
| 01 | Preprocessing | Normalisasi slang, konversi BIEOS, filtering |
| 02 | Tokenization | Tokenisasi IndoBERT, label alignment, prior embedding |
| 03 | MultiTask_Training | Training multi-task (F1=0.7410) |
| 04 | Hyperparameter_Search | Grid search λ, dropout, LR |
| 05 | Data_Augmentation | Augmentasi data → 4,290 samples |
| 06 | MultiTask_Augmented | MT + augmented data + freeze layers |
| 07 | SingleTask_Baseline | Baseline IndoBERT+CRF (F1=0.7551) |
| 08 | Lambda_Phase_Tuning | Tuning λ dan konfigurasi phase |
| 09 | MultiSeed_Robustness | Robustness test (5 seeds) |
| 10 | Extended_Training_50ep | 4 runs × 50 epochs → **best F1=0.7600** |
| 11 | Ablation_Study | Softmax vs CRF vs MTL+CRF |
| 12 | Base_Model_Comparison | IndoBERT-large vs alternatives |
| 13 | MTL_Architecture_Variants | Label smoothing, task dropout, partial sharing |
| 14 | Evaluation_ErrorAnalysis | Confusion matrix, error analysis, robustness |
| 15 | Visualization_Interpretability | Learning curves, attention heatmap, t-SNE |
| 16 | MTL_NoSLD | Ablation: remove SLD subtask |
| 17 | ClassWeighted_MTL | Class-weighted loss untuk O-label dominance |
| 18 | Embedding_Analysis | Analisis embedding space MT vs ST |
| 19 | CrossAttention_O_Cluster_Explanation | Cross-attention & O-token clustering |
| 20 | MTL_Advantage_Analysis | Per-class F1, multi-aspect, distance analysis MT vs ST |

### Sentiment Analysis Pipeline

Notebooks di folder `sentiment analysis/` untuk scraping, cleaning, dan dashboard analisis sentimen desa wisata.

| Notebook | Deskripsi |
|----------|-----------|
| Scrape_GoogleMaps | Scraping review Google Maps via Apify API |
| Clean_Reviews | Cleaning: slang normalization, stopwords removal (Sastrawi), stemming |
| SA_AutoLabel | Auto-labeling sentimen (IndoBERT) + aspect detection + export JSON |

**Flow:** `Scrape_GoogleMaps` → `Clean_Reviews` → `SA_AutoLabel` → Dashboard

Dashboard (Dash app) di `sentiment analysis/dashboard/app.py`.

## Struktur Proyek

```
├── data/
│   ├── raw/                          # Dataset ABSA asli + lexicon
│   ├── processed/                    # Data tokenized siap training
│   └── raw_review/
│       ├── *.txt                     # Raw review 6 desa awal
│       ├── Overall_All Data_for_Prediction.xlsx
│       └── additional/
│           ├── raw/                  # Hasil scraping mentah (*_raw.csv)
│           └── cleaned/             # Hasil cleaning (*_reviews.csv, *_full.csv)
├── models/                           # Model weights & checkpoints
├── notebooks/                        # ABSA notebooks (00-20)
├── sentiment analysis/
│   ├── Scrape_GoogleMaps.ipynb       # Scraping via Apify
│   ├── Clean_Reviews.ipynb           # Cleaning & preprocessing
│   ├── SA_AutoLabel.ipynb            # Auto-labeling + aspect detection
│   ├── dashboard/                    # Dash web app
│   └── data/                         # Output JSON + CSV untuk dashboard
└── emc gcn/                          # EMC-GCN baseline (prior research)
```

## Referensi

- Wang et al. (2021) — Hierarchical Multi-Task Learning for ABSA
- Koto et al. (2020) — IndoBERT
- Li et al. (2019) — Unified ABSA tagging
- He et al. (2019) — Interactive MTL for ABSA
