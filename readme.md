#  CheXNet-Style Multi-Label Chest X-Ray Classification (PyTorch)

This project implements a **CheXNet-inspired deep learning pipeline** for **multi-label classification of chest X-ray images** using **DenseNet-121** in PyTorch.

The model predicts **multiple thoracic diseases simultaneously** from a single chest X-ray image using transfer learning and AUROC-based evaluation.

---

##  Key Features

- Multi-label disease classification
- DenseNet-121 pretrained on ImageNet
- Custom PyTorch Dataset for NIH ChestX-ray14
- AUROC evaluation per disease
- Checkpointing & resume training
- CUDA / CPU support

---

##  Disease Labels

The model supports the following conditions:

- Atelectasis  
- Consolidation  
- Infiltration  
- Pneumothorax  
- Edema  
- Emphysema  
- Fibrosis  
- Effusion  
- Pneumonia  
- Pleural_Thickening  
- Cardiomegaly  
- Nodule  
- Hernia  
- Mass  
- No Finding  
- Other  

Each image can contain **multiple labels**.

---

##  Dataset

Designed for the **NIH ChestX-ray14 dataset**.

### Directory Structure

dataset_root/
├── images_001/
│ └── images/
│ ├── 00000001_000.png
│ └── ...
├── images_002/
│ └── images/
│ └── ...
└── Data_Entry_2017.csv


### CSV Fields Used

- `Image Index` → Image filename
- `Finding Labels` → Pipe-separated disease names

Example:  00000001_000.png, Cardiomegaly|Edema


---

##  Data Preprocessing

### Percent Center Crop
A custom transform that crops a percentage from the center of the image to remove borders.

### Transform Pipeline

- Center crop
- Resize to 256×256
- Convert to tensor
- Normalize using ImageNet statistics

This ensures compatibility with pretrained DenseNet models.

---

##  Custom Dataset (`ChestXRayDataSet`)

Responsibilities:
- Reads image paths from multiple subfolders
- Parses labels from CSV
- Converts labels to **multi-hot vectors**
- Dynamically builds class-to-index mapping

Example label tensor:[0, 1, 0, 0, 1, 0, ...]


---

##  Model Architecture

### Backbone
- DenseNet-121 (ImageNet pretrained)

### Classifier Head
The default classifier is replaced with: Linear(1024 → num_classes)


The model outputs **raw logits** (sigmoid applied during evaluation).

---

##  Training Configuration

| Parameter        | Value        |
|------------------|-------------|
| Optimizer        | AdamW       |
| Learning Rate    | 1e-4        |
| Weight Decay     | 1e-4        |
| Loss Function    | BCEWithLogitsLoss |
| Batch Size       | 12          |
| Epochs           | 10          |

---

##  Loss Function

**BCEWithLogitsLoss** is used because:
- It supports multi-label classification
- It is numerically stable
- It combines sigmoid + binary cross-entropy

---

##  Evaluation Metric

Uses **Multilabel AUROC** from `torchmetrics`.

During validation:
- Sigmoid is applied to logits
- AUROC is computed per disease
- Mean AUROC is reported

Example: 
Validation AUROC per class:
Atelectasis           : 0.8655
Cardiomegaly          : 0.9630
Consolidation         : 0.8625
Edema                 : 0.9245
Effusion              : 0.9207
Emphysema             : 0.9331
Fibrosis              : 0.8593
Hernia                : 0.9968
Infiltration          : 0.7670
Mass                  : 0.8926
No Finding            : 0.8220
Nodule                : 0.8062
Pleural_Thickening    : 0.8387
Pneumonia             : 0.7903
Pneumothorax          : 0.9444

Mean AUROC: 0.8791
Val Loss  : 0.1569


---

##  Training Loop

### Training Phase
- Forward pass
- Loss computation
- Backpropagation
- Optimizer step
- Batch-wise loss logging

### Validation Phase
- No gradient computation
- AUROC accumulation
- Per-class and mean AUROC reporting

---

##  Checkpointing

Automatically saves:
- Model weights
- Optimizer state
- Best validation AUROC
- Epoch number

Training resumes automatically if a checkpoint exists.


##  Best Model Saving

The best-performing model (based on validation AUROC) is saved as:best_chexnet_params.pth


---

##  How to Run
Terminal-->
python train.py

This project is for research and educational purposes only.
It is not approved for clinical use.

## Acknowledgements

NIH ChestX-ray14 Dataset

CheXNet (Rajpurkar et al.)

PyTorch & Torchvision








