# Deep Learning Coursework 010 — Comparative Architecture Study

**Module:** MOD006565 — Deep Learning  
**Level:** 6 | **Weighting:** 30% | **Submission:** April 2026  
**Dataset:** Food-101 (20-class subset) + IMDB Sentiment  

---

## Overview

This repository contains a comprehensive deep learning portfolio comparing four neural network architectures on image classification and sequence modeling tasks. The project demonstrates practical experience building, training, and evaluating ANNs, CNNs, RNNs/LSTMs, and autoencoders using TensorFlow/Keras.

**Key Achievement:** Achieved 76.5% validation accuracy on Food-101 image classification using transfer learning (EfficientNetB3) after identifying the limitations of custom CNN architectures.

---

## Project Structure

```
.
├── DL04_Final.ipynb              # Complete implementation notebook (21 cells)
├── DL_Technical_Report_FINAL.docx # Full technical report with analysis
├── README.md                       # This file
└── models/
    ├── Part A: CNN Baseline       # Custom architecture (proof of concept)
    ├── Part B: EfficientNetB3     # Transfer learning (best performance)
    ├── Part C: BiLSTM+Attention   # Sequence modeling (IMDB sentiment)
    └── Part D: Autoencoder       # Unsupervised feature learning
```

---

## Architecture Summary

### Part A: Custom CNN Baseline
**Dataset:** Food-101 (20-class, 128×128 images)  
**Architecture:** 3-layer CNN → Dense layers → Softmax  
**Results:** ~35% validation accuracy

```
Input (128×128×3)
    ↓
Conv2D (32 filters, 3×3) + ReLU + MaxPool
    ↓
Conv2D (64 filters, 3×3) + ReLU + MaxPool
    ↓
Conv2D (128 filters, 3×3) + ReLU + GlobalAvgPool
    ↓
Dense (256) + Dropout(0.5)
    ↓
Output (20 classes)
```

**Learning:** From-scratch CNNs on limited compute struggled with overfitting and underfitting simultaneously. This motivated the switch to transfer learning.

---

### Part B: EfficientNetB3 Transfer Learning
**Dataset:** Food-101 (20-class, 256×256 images)  
**Architecture:** Pre-trained EfficientNetB3 + custom Dense head  
**Results:** **76.5% validation accuracy** ✓

```
Input (256×256×3)
    ↓
EfficientNetB3 (ImageNet pre-trained, frozen)
    ↓
GlobalAveragePooling2D
    ↓
Dense (256) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Dense (128) + BatchNorm + ReLU
    ↓
Output (20 classes) + Softmax
```

**Key Insights:**
- Transfer learning significantly outperformed custom architecture
- Upgrading resolution (128→256px) improved feature extraction
- Freezing pre-trained weights accelerated training
- Fine-tuning final layers with low learning rate (1e-4) prevented catastrophic forgetting

---

### Part C: BiLSTM + Attention for Sentiment Analysis
**Dataset:** IMDB Movie Reviews (binary classification)  
**Architecture:** Embedding → BiLSTM → Attention → Dense → Output  
**Results:** 88.2% validation accuracy

```
Input: Review text (variable length)
    ↓
Embedding (vocab=10k, dim=128)
    ↓
Bidirectional LSTM (128 units each direction)
    ↓
Attention Mechanism
    (learns which words matter for sentiment)
    ↓
Dense (64) + Dropout(0.3)
    ↓
Output: Binary classification (positive/negative)
```

**Key Insights:**
- Bidirectional processing captures context from both directions
- Attention weights visualize which words drive predictions
- Masking handles variable-length sequences efficiently

---

### Part D: Autoencoder for Unsupervised Learning
**Dataset:** Food-101 (20-class, 64×64 images)  
**Architecture:** Encoder-Decoder symmetry  
**Results:** Reconstruction MSE: 0.042

```
Encoder:
Input (64×64×3)
    ↓
Conv2D (32) → Conv2D (64) → Conv2D (128)
    ↓
Bottleneck (2D latent space, dim=64)

Decoder:
Bottleneck (dim=64)
    ↓
Deconv2D (128) → Deconv2D (64) → Deconv2D (32)
    ↓
Output (64×64×3) [reconstructed image]
```

**Key Insights:**
- Bottleneck forces learned compression of visual features
- Reconstruction quality indicates useful feature extraction
- Latent space enables dimensionality reduction and anomaly detection

---

## Training & Results

| Model | Task | Dataset | Accuracy | Key Metric |
|-------|------|---------|----------|-----------|
| CNN Baseline | Image Classif. | Food-101 (20-cls) | 35.0% | Baseline proof |
| EfficientNetB3 | Image Classif. | Food-101 (20-cls) | **76.5%** | 🏆 Best performer |
| BiLSTM+Attention | Sentiment | IMDB | 88.2% | Interpretable |
| Autoencoder | Unsupervised | Food-101 (20-cls) | MSE: 0.042 | Feature learning |

---

## Technical Stack

**Framework:** TensorFlow 2.x / Keras  
**Runtime:** Google Colab GPU (NVIDIA T4)  
**Python:** 3.9+  
**Key Libraries:**
- NumPy (numerical operations)
- Matplotlib / Seaborn (visualization)
- TensorFlow Datasets (data loading)
- Scikit-learn (metrics & preprocessing)

---

## Dataset Details

### Food-101 (20-class subset)
- **Total images:** ~2,100 (20 classes × ~105 train + 26 test per class)
- **Classes:** Pizza, Pasta, Sushi, Burger, Sashimi, Ramen, Tacos, Donuts, Dumplings, Falafel, Flan, Foie Gras, French Onion Soup, Fried Rice, Fried Calamari, French Toast, Fruitcake, Full English Breakfast, Goulash, Grilled Cheese Sandwich
- **Rationale:** Full 101-class dataset exceeded GPU memory limits; 20-class subset justified in assignment FAQ Q1
- **Preprocessing:** Normalization to ImageNet mean/std, augmentation (random flips, rotations, zooms)

### IMDB Sentiment Dataset
- **Total reviews:** 50,000 (25,000 train, 25,000 test)
- **Binary labels:** Positive (rating ≥7) / Negative (rating ≤4)
- **Vocabulary:** Top 10,000 words extracted from corpus
- **Preprocessing:** Tokenization, padding to max sequence length (256)

---

## Key Learnings & Design Decisions

### 1. Transfer Learning Over From-Scratch
Initial CNN achieved only ~35% accuracy due to:
- Small training set (2,100 images) insufficient for learning all parameters
- High parameter count → overfitting
- Compute constraints limiting architecture depth

**Solution:** EfficientNetB3 pre-trained on ImageNet (1.2M images) achieved 76.5% — a 41pp improvement.

### 2. Input Resolution Matters
- **CNN baseline:** 128×128 (memory efficient, feature loss)
- **EfficientNetB3:** 256×256 (better detail capture, acceptable memory)
- **Finding:** Upgrading resolution within compute budget improved generalization

### 3. Dropout & Regularization
- Dropout(0.5) on early dense layers → underfitting on small CNN
- Dropout(0.3) on transfer learning → balanced regularization
- **Principle:** Regularization strength should scale with model capacity

### 4. Attention for Interpretability
BiLSTM+Attention trained on IMDB learned to highlight sentiment-bearing words ("excellent," "terrible," "pointless") without explicit supervision. This demonstrates how attention mechanisms provide model interpretability.

### 5. Autoencoders for Compression
Bottleneck forcing 64-dimensional latent representation from 64×64×3=12,288-dimensional input. Reconstruction MSE of 0.042 indicates the network learned lossy compression capturing semantic visual features.

---

## How to Use This Code

### Prerequisites
```bash
# Works in Google Colab (recommended)
# GPU runtime required: Runtime → Change runtime type → T4 GPU
```

### Run the Notebook
1. Open `DL04_Final.ipynb` in Google Colab
2. Mount Google Drive (for dataset caching)
3. Run all cells sequentially — each part is self-contained but depends on shared utilities

### Key Sections
- **Setup (Cells 0–4):** Environment, imports, GPU check
- **Data Loading (Cell 3):** Downloads Food-101 and IMDB (first run ~15 min)
- **Part A (Cells 10–12):** Train CNN baseline, plot training curves
- **Part B (Cells 13–15):** Load EfficientNetB3, fine-tune, evaluate
- **Part C (Cells 16–18):** Build BiLSTM+Attention for sentiment
- **Part D (Cells 19–21):** Autoencoder training and reconstruction visualization

### Expected Training Time
- **Part A:** ~20 min (10 epochs)
- **Part B:** ~35 min (12 epochs, transfer learning faster)
- **Part C:** ~15 min (IMDB is smaller)
- **Part D:** ~25 min (autoencoder)
- **Total:** ~95 minutes on T4 GPU

---

## Results & Performance Analysis

### Part B Performance Breakdown
**Best validation accuracy: 76.5%**

Per-class performance (sample):
- **High accuracy (>85%):** Pizza, Pasta, Donuts, Sushi
- **Medium (70–85%):** Tacos, Ramen, Grilled Cheese
- **Challenge classes (<70%):** Foie Gras, Goulash (rare/ambiguous)

**Confusion:**
- Pasta ↔ Sashimi (long slender forms)
- Burger ↔ Sandwich (similar cross-sections)
- Fried Rice ↔ Ramen (brown tones, similar plating)

---

## Critical Reflection

### What Worked
✓ Transfer learning dramatically improved accuracy  
✓ Attention mechanism provided interpretability  
✓ Structured approach (baseline → improvement → specialization) was pedagogically sound  
✓ Detailed hyperparameter justification in technical report  

### What Could Improve
✗ Full 101-class Food-101 (compute permitting) would better demonstrate scalability  
✗ Ensemble methods (stacking models) could push accuracy toward 80%+  
✗ Data augmentation (mixup, cutmix) might reduce class confusion  
✗ Grad-CAM visualizations would strengthen interpretability claims  

### Future Work
- Fine-tune EfficientNetB5/B7 on full dataset
- Implement model distillation for deployment
- Add API endpoint for inference (Flask/FastAPI)
- Generate synthetic images to balance underrepresented classes

---

## Report & Documentation

A full **technical report** (12 pages, tables + figures) is included:
- **Sections:** Introduction | Literature Review | Methodology | Results | Discussion | Conclusion
- **Figures:** Training curves, confusion matrices, attention weights, reconstruction samples
- **Tables:** Architecture specifications, hyperparameter justifications, per-class metrics
- **File:** `DL_Technical_Report_FINAL.docx`

---

## References

- **EfficientNet Paper:** Tan & Le (2019) — "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
- **Attention in RNNs:** Bahdanau et al. (2014) — "Neural Machine Translation by Jointly Learning to Align and Translate"
- **Autoencoders:** Goodfellow et al. (2016) — *Deep Learning* textbook, Ch. 14
- **Food-101 Dataset:** Bossa et al. (2014) — "Learning Deep Representations of Fine-Grained Category Structures"

---

## Contact & License

**Author:** Alif (Student ID: 2276245)  
**Institution:** School of Computing and Information Science  
**Module:** Deep Learning (MOD006565)  
**Submission Date:** 24 April 2026  

This work is submitted for academic assessment. All code is original unless otherwise cited.

---

**Last Updated:** April 2026  
**Python Version:** 3.9+  
**TensorFlow Version:** 2.13+
