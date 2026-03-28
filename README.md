# DeepLearningProject_Let_Samples_Speak
Reproduction &amp; extension of NSF (CVPR 2025) — a bias debiasing framework that neutralizes spurious correlations in deep learning models without requiring bias labels. Tested on Waterbirds, CelebA, and CIFAR-10 on Google Colab.


# 🧠 NSF Debiasing — CVPR 2025 Reproduction

> **Let Samples Speak: Mitigating Spurious Correlation by
> Exploiting the Clusterness of Samples**
> *CVPR 2025 — Reproduction & Extension*

[![Paper](https://img.shields.io/badge/Paper-CVPR%202025-blue)](https://cvpr.thecvf.com/virtual/2025/poster/34895)
[![Original Repo](https://img.shields.io/badge/Original-GitHub-black)](https://github.com/davelee-uestc/nsf_debiasing)
[![YouTube](https://img.shields.io/badge/Demo-YouTube-red)](https://youtu.be/hYH4vwrVTmI)
[![Colab](https://img.shields.io/badge/Run%20on-Google%20Colab-orange)](https://colab.research.google.com)

---

## 👥 Team

| Name | Roll No |
|------|---------|
| Azmeera Ravindar Naik | B22CS017 |
| L. Umesh Chandra | B22CS029 |
| Y. Varun Kumar | B22CS065 |

---

## 📌 What is NSF?

Deep learning models often learn **spurious correlations** —
shortcuts in training data that happen to correlate with the
label but are causally irrelevant. For example:

- A bird classifier learns **background** (water/land) instead
  of the bird itself
- A hair color classifier learns **gender** instead of actual
  hair color
- A medical AI detects **equipment** instead of disease symptoms

**NSF (Neutralizing Spurious Features)** fixes this
automatically — without needing to know what the bias is,
and without requiring any bias labels.

### How it works (4 steps):
```
ERM Trained Model
       ↓
1. IDENTIFY  → Find minority samples that deviate from class centroid
       ↓
2. NEUTRALIZE → Estimate true unbiased class means
       ↓
3. ELIMINATE  → Learn transformation t(x) to suppress spurious channels
       ↓
4. UPDATE     → Fine-tune classifier on bias-free features
```

---

## 🏆 Our Results

| Dataset | Before (ERM WGA) | After (NSF WGA) | Improvement |
|---------|-----------------|-----------------|-------------|
| **Waterbirds** | 72.4% | **83.5%** | +11.1% ✅ |
| **CelebA** | 41.7% | **65.0%** | +23.3% ✅ |
| **CIFAR-10** | 90.7% | 89.5% | -1.2% (no bias — expected) |

> 📝 Trained with 5–10 epochs due to Colab limits.
> Paper reports 91.1% (Waterbirds) and 84.3% (CelebA) with full training.

---

## 📁 Repository Structure
```
nsf_debiasing/
│
├── train_supervised.py     # Step 1: Train base ERM model
├── ssc.py                  # Step 2: Run NSF debiasing (main method)
├── ssc_common.py           # Shared utilities
├── train_sup.sh            # Shell script (bypassed in our setup)
│
├── data/
│   └── datasets.py         # Dataset classes (Waterbirds, CelebA, etc.)
├── models/                 # ResNet-50, BERT model wrappers
├── utils/                  # Logging, training utilities
├── optimizers/             # SGD, AdamW, BERT optimizer
└── dataset_files/          # Metadata helpers
```

---

## 🚀 Quick Start (Google Colab)

### Step 1 — Setup
```python
# Clone and install
!git clone https://github.com/davelee-uestc/nsf_debiasing.git
%cd nsf_debiasing
!pip install timm transformers scikit-learn tqdm wilds -q
```

### Step 2 — Apply Patches
```python
# Fix mmcv (not compatible with Python 3.12)
with open('ssc_common.py', 'r') as f:
    content = f.read()
content = content.replace('import mmcv', 'from tqdm import tqdm')
content = content.replace('prog=mmcv.ProgressBar(len(loader))',
                          'prog=tqdm(total=len(loader))')
content = content.replace('prog=mmcv.ProgressBar(N)',
                          'prog=tqdm(total=N)')
with open('ssc_common.py', 'w') as f:
    f.write(content)

# Disable wandb
with open('train_supervised.py', 'r') as f:
    content = f.read()
content = content.replace(
    'try:\n    import wandb\n    has_wandb = True\n'
    'except ImportError:\n    has_wandb = False',
    'has_wandb = False'
)
with open('train_supervised.py', 'w') as f:
    f.write(content)

print("✅ Patches applied!")
```

### Step 3 — Download Dataset

**Waterbirds:**
```python
!wget -q "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz" \
    -O /content/waterbirds.tar.gz
!mkdir -p /content/nsf_debiasing/waterbirds
!tar -xzf /content/waterbirds.tar.gz \
    -C /content/nsf_debiasing/waterbirds/ --strip-components=1
```

**CelebA** (requires Kaggle API):
```python
!kaggle datasets download -d jessicali9530/celeba-dataset

import zipfile, os
from tqdm import tqdm
dst = '/content/nsf_debiasing/celeba/img_align_celeba'
os.makedirs(dst, exist_ok=True)
with zipfile.ZipFile('celeba-dataset.zip', 'r') as z:
    imgs = [f for f in z.namelist() if f.endswith('.jpg')]
    for f in tqdm(imgs):
        fname = os.path.basename(f)
        with z.open(f) as s, open(os.path.join(dst, fname), 'wb') as o:
            o.write(s.read())

# Build metadata.csv
import pandas as pd
attr = pd.read_csv('/content/nsf_debiasing/celeba/list_attr_celeba.csv')
part = pd.read_csv('/content/nsf_debiasing/celeba/list_eval_partition.csv')
df = attr.merge(part, on='image_id')
df['y'] = (df['Blond_Hair'] == 1).astype(int)
df['spurious'] = (df['Male'] == 1).astype(int)
df['split'] = df['partition']
df['img_filename'] = 'img_align_celeba/' + df['image_id']
df[['img_filename','y','spurious','split']].to_csv(
    '/content/nsf_debiasing/celeba/metadata.csv', index=False)
```

### Step 4 — Train Base ERM Model
```python
# Waterbirds
!CUDA_VISIBLE_DEVICES=0 python3 train_supervised.py \
    --output_dir=/content/nsf_debiasing/logs/waterbirds/erm_seed1 \
    --num_epochs=10 --eval_freq=1 --save_freq=10 --seed=1 \
    --weight_decay=1e-4 --batch_size=32 --init_lr=3e-3 \
    --scheduler=cosine_lr_scheduler \
    --data_dir=/content/nsf_debiasing/waterbirds \
    --data_transform=AugWaterbirdsCelebATransform \
    --dataset=SpuriousCorrelationDataset \
    --model=imagenet_resnet50_pretrained

# CelebA
!CUDA_VISIBLE_DEVICES=0 python3 train_supervised.py \
    --output_dir=/content/nsf_debiasing/logs/celeba/erm_seed1 \
    --num_epochs=5 --eval_freq=1 --save_freq=1 --seed=1 \
    --weight_decay=1e-4 --batch_size=100 --init_lr=3e-3 \
    --scheduler=cosine_lr_scheduler \
    --data_dir=/content/nsf_debiasing/celeba \
    --data_transform=AugWaterbirdsCelebATransform \
    --dataset=SpuriousCorrelationDataset \
    --model=imagenet_resnet50_pretrained
```

### Step 5 — Run NSF Debiasing
```python
!python3 ssc.py waterbirds
!python3 ssc.py celeba
!python3 ssc.py cifar10   # our extension
```

---

## 🔧 Our Extensions & Fixes

| Issue | Fix |
|-------|-----|
| `mmcv` incompatible with Python 3.12 | Replaced with `tqdm` |
| `vissl` install fails | Removed — never imported |
| `wandb` blocks Colab with login prompt | Disabled via `has_wandb = False` |
| `train_sup.sh` fails on Colab GPU check | Bypassed, run Python directly |
| Relative paths fail on Colab | Replaced with absolute paths |
| CelebA has no `metadata.csv` | Built from `list_attr_celeba.csv` |
| CelebA nested zip extraction | Used Python `zipfile` with `basename()` |
| CIFAR-10 not in `ssc.py` | Added with `n_classes=10` fix |

---

## 📊 Dataset Details

| Dataset | Task | Spurious Feature | Train Size | Groups |
|---------|------|-----------------|------------|--------|
| Waterbirds | Bird species | Background | 4,795 | 4 |
| CelebA | Hair color | Gender | 162,770 | 4 |
| CIFAR-10 | Object class | None (baseline) | 45,000 | 10 |

---

## ⚠️ Common Issues

**Q: `ModuleNotFoundError: No module named 'mmcv'`**
A: Run the patch in Step 2 above.

**Q: `FileNotFoundError: logs/waterbirds/erm_seed1/final_checkpoint.pt`**
A: You need to run Step 4 (train ERM) before Step 5 (NSF debiasing).

**Q: `AssertionError: Torch not compiled with CUDA enabled`**
A: Go to Runtime → Change runtime type → T4 GPU → Save.

**Q: `KeyError: 'cifar10'` in ssc.py**
A: CIFAR-10 is our extension. Apply the patch in Step 2 or
add it manually to `all_args_map` in `ssc.py`.

**Q: Colab session disconnects during training**
A: Use `save_freq=1` and run this auto-backup in a parallel cell:
```python
import time, shutil
while True:
    time.sleep(1500)
    shutil.copytree('logs', '/content/drive/MyDrive/backup_logs',
                    dirs_exist_ok=True)
```

---

## 📚 Citation

If you use this work, please cite the original paper:
```bibtex
@inproceedings{li2024let,
  title={Let Samples Speak: Mitigating Spurious Correlation
         by Exploiting the Clusterness of Samples},
  author={Weiwei Li and Junzhuo Liu and Yuanyuan Ren and
          Yuchen Zheng and Yahao Liu and Wen Li},
  booktitle={Proceedings of the IEEE/CVF Conference on
             Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```

---

## 📄 License

This reproduction follows the original repository's
GPL-3.0 license. See [LICENSE](LICENSE) for details.
