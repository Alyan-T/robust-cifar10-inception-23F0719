# Robust CIFAR-10 Classification with Limited Data

**Course:** Deep Learning for Perception (Fall 2026), FAST-NUCES
**Assignment:** 2, Section B
**Name:** Alyan Tariq
**Roll No:** 23F-0719

## About
An Inception-style CNN (about 63K parameters) trained from scratch on only 500 CIFAR-10 images (400 train / 100 validation). It is evaluated on the clean CIFAR-10 test set and on a simulated shifted test set (rotation, brightness/contrast change, occlusion and a hidden grayscale corruption). The notebook covers the baseline, augmentation strategies, regularization, an ablation study, error analysis and test-time augmentation.

## Files
| File | Description |
|---|---|
| `DLP_AS2_23F-0719_7D_SectionB.ipynb` | Executed notebook (Parts A to F) with all outputs and plots |
| `README.md` | This file |

## Requirements
- Python 3.9 or newer
- `tensorflow` (Keras), `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `scipy`
- GPU recommended. The notebook was run on Kaggle with a Tesla T4.

## How to reproduce the results
1. Upload the notebook to Kaggle (or Colab/Jupyter).
2. Add the CIFAR-10 python dataset (`pankrzysiu/cifar10-python`) on Kaggle. The notebook reads it from `/kaggle/input/datasets/pankrzysiu/cifar10-python/cifar-10-batches-py`. If you run elsewhere, change `DATA_DIR` in the first cell to the folder that contains `data_batch_1` ... `data_batch_5` and `test_batch`.
3. Run all cells from top to bottom.

## Seeds and settings
- Training seeds: `10, 20, 30` (set with `tf.keras.utils.set_random_seed`). Results are reported as mean +- std over these 3 seeds.
- Train/validation split: `train_test_split(..., test_size=100, random_state=42, stratify=...)` gives 400 train / 100 validation images (40 / 10 per class).
- Shifted test set: generated with `np.random.default_rng(42)`.
- Hidden-corruption test set: generated with `np.random.default_rng(2026)`.
- Maximum 60 epochs. Batch size 32. Adam optimizer.
- The 500-image subset is drawn with `np.random.choice`. Call `np.random.seed(...)` before that cell if you need exactly the same 500 images on every run.

## Rules followed
- No pretrained weights (trained from scratch).
- Fewer than 1,000,000 trainable parameters (63,218 for the baseline model).
- Includes an Inception-style block (parallel 1x1, 1x1->3x3 and 1x1->5x5 branches, concatenated).
- Augmentation is applied on the fly inside the model, so it only affects training batches.
- Hyperparameters and model choice were based on the validation split and the 3-seed results.

## Model selection (Part A)
Three Inception-style models were compared without augmentation (mean over 3 seeds):

| Model size | Val Acc | Clean Test | Shifted Test |
|---|---|---|---|
| ~28K parameters | 32.67% | 33.76% | 17.12% |
| ~64K parameters (chosen baseline) | 40.00% | 36.35% | 16.80% |
| ~228K parameters | 39.00% | 34.44% | 16.23% |

The 64K model was chosen as the baseline because it had the best validation and clean-test accuracy.

## Results (mean +- std over 3 seeds)

| Configuration | Val Acc | Clean Test | Shifted Test |
|---|---|---|---|
| No augmentation (baseline) | 40.00% +- 2.00% | 36.35% +- 1.43% | 16.80% +- 1.17% |
| Geometric only (flip, rotation, translation) | 36.67% +- 5.13% | 33.11% +- 3.61% | 18.96% +- 1.51% |
| Photometric only (brightness, contrast) | 35.33% +- 3.51% | 36.33% +- 1.01% | **21.82% +- 1.19%** |
| Geometric + photometric | 36.67% +- 3.21% | 34.26% +- 2.06% | 21.29% +- 2.17% |
| Regularized model (Part C: BatchNorm, L2, Dropout, label smoothing, early stopping, LR schedule) | 12.33% +- 1.53% | 12.32% +- 1.29% | 11.39% +- 1.31% |

### Summary
- **Best shifted-test score:** 21.82% +- 1.19% (photometric-only augmentation).
- **Configuration behind it:** ~63K-parameter Inception CNN with random brightness (0.15) and random contrast (0.20) applied on the fly.
- **Single change that helped most:** adding photometric augmentation, which gave +5.02 percentage points on the shifted test (16.80% to 21.82%). It matches the brightness/contrast change in the shifted test set.
- **Baseline diagnosis:** train accuracy (55.4%) is above validation (40.0%), so the model overfits. Clean-to-shifted accuracy falls from 36.35% to 16.80%, which shows covariate shift as well.
- **Part C regularized model:** it stopped after 8-11 epochs and predicted almost every test image as one class (mostly "airplane"), so it scored lower than the baseline.
- **Error analysis (Part E):** per-class accuracy, confusion matrix and misclassified images were produced for the Part C model on the clean test set. Because that model mostly predicts "airplane", the confusion matrix shows one dominant column.
- **Hidden corruption (grayscale):** the Part C model scored 10.86% clean, 10.42% on a 50/50 mix of grayscale and colour, and 9.65% on grayscale images only.
- **Test-time augmentation (bonus):** averaging the original, a horizontal flip and a small shift gave +0.00 points (11.02% to 11.02%) at about 1.22x the inference time.
