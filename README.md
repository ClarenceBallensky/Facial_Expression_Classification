# Facial Expression Classification

Deep learning model created for the [Facial Expression Classification](https://www.kaggle.com/competitions/cs-4210-fall-2026-assignment-3-facial-expression-classification/submissions#) competition on Kaggle.
Reference `research_poster.pdf` for implementation details. 

## Results

| **Metric** | **Validation (with TTA)** |
|:---|:---|
| Accuracy | `0.836` |
| Macro precision | `0.821` |
| Macro recall | `0.819` |
| Macro F1 | `0.819` |

Kaggle test accuracy: `0.837`

<img width="450" alt="image" src="https://github.com/user-attachments/assets/fa49b9f5-8a35-45cf-832b-d0a9ec5a482a" />

Note: runs are not bit-for-bit reproducible across hardware and library versions, so expect small differences (a few tenths of a point) from the numbers above.

## Settings used

| **Setting** | **Value** |
|:--- |:--- |
| Random seed | 42 (`SEED`), split `random_state=42` |
| Data split | 80/20 train/validation, stratified by class |
| Input | 48×48 grayscale scaled to 0–1, resized to 96×96 |
| Model | ResNet-50 (1-channel `conv1`), pretrained weights remapped from `final_model.pth`, new head: Dropout(0.5) + Linear(2048, 3) |
| Trained layers | `layer3`, `layer4`, and the new head (earlier layers frozen, kept in eval mode) |
| Optimizer | AdamW, weight decay 0.02; learning rates `layer3` 3e-5, `layer4` 7e-5, head 1e-4 |
| Schedule | Cosine annealing over 20 epochs |
| Loss | Cross-entropy with label smoothing 0.1 |
| Batch size | 64, plain shuffling |
| Training augmentation | Random horizontal flip, rotation up to 10°, shift up to 10%, scale 0.9–1.1 |
| Model selection | Best validation accuracy across epochs |
| Inference | Flip test-time augmentation (average of image and mirrored image) |

## Requirements

- Python 3.10+ (developed on `<your Python version>`)
- Packages (install with `pip install -r requirements.txt`):

```text
torch>=2.6
torchvision>=0.21
numpy
pandas
scikit-learn
matplotlib
jupyter
```

## Files needed

The files `train_data.csv`, `train_target.csv`, and `test_data.csv` are too large to commit to GitHub. Download them from the `Data` section of the [Facial Expression Classification](https://www.kaggle.com/competitions/cs-4210-fall-2026-assignment-3-facial-expression-classification/submissions#) competition on Kaggle.

```text
.
├── code.ipynb                  # the notebook
├── base_model.pth              # pretrained ResNet-50 checkpoint (source: <link or where you got it>)
└── data/
    ├── train_data.csv          # training images: one row per image, 2304 pixel values (48x48), no header
    ├── train_target.csv        # training labels: one column, no header
    └── test_data.csv           # test images, same format as train_data.csv, no labels provided
```

Labels: `0` = angry, `1` = happy, `2` = neutral.

## How to run

1. Install the requirements.
2. Install the necessary data files. 
3. Open `code.ipynb` in Jupyter Notebook.
4. **Restart the kernel and run all cells in order** (the model and optimizer are created in specific cells, so rerunning individual cells continues from already-trained weights).
5. After training, the notebook writes `submission_full.csv` (columns `Id`, `Category`) and saves the best weights to `model.pth`.
