# Federated Learning for COVID-19 and Pneumonia Chest X-ray Classification

This repository implements a federated learning pipeline for classifying chest X-ray images into COVID-19, Pneumonia, and Normal categories. The project uses PyTorch, Opacus for differential privacy, and supports data augmentation and class balancing.

## Folder Structure

```
covid19.zip
pneumonia.zip
evaluation.py
metrics_log.csv
metrics_log1.csv
test.ipynb
test_densenet121.ipynb
covid19/
    COVID-19_Radiography_Dataset/
        COVID.metadata.xlsx
        Lung_Opacity.metadata.xlsx
        Normal.metadata.xlsx
        README.md.txt
        Viral Pneumonia.metadata.xlsx
        COVID/
        Lung_Opacity/
        Normal/
        Viral_Pneumonia/
pneumonia/
    chest_xray/
        __MACOSX/
        chest_xray/
        test/
        train/
        val/
federated_metrics_plots/
    federated_metrics_plots/
        comparison/
        densenet121/
        resnet18/
```

## Main Files

- [`test.ipynb`](test.ipynb): Main notebook for federated learning with ResNet18.
- [`test_densenet121.ipynb`](test_densenet121.ipynb): Notebook for federated learning with DenseNet121.
- [`evaluation.py`](evaluation.py): Script for evaluating trained models.
- [`metrics_log.csv`](metrics_log.csv): Training and evaluation metrics log.

## Datasets

- **COVID-19 Radiography Dataset**: Located in `https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database`
- **Chest X-ray Pneumonia Dataset**: Located in `https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia`

## Features

- Federated learning with intermittent client participation
- Differential privacy using Opacus
- Data augmentation and class balancing (SMOTE, oversampling)
- Support for ResNet18 and DenseNet121 backbones
- Per-client and global model evaluation
- Metrics logging and plotting

## Requirements

- Python 3.10+
- PyTorch 2.0.1
- torchvision 0.15.2
- torchaudio 2.0.2
- opacus 1.3.0
- fedml
- scikit-learn
- pandas
- imbalanced-learn
- matplotlib
- seaborn
- tqdm

Install dependencies using:

```sh
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install fedml scikit-learn pandas imbalanced-learn matplotlib seaborn tqdm opacus==1.3.0
```

## Usage

1. Unzip the datasets (`covid19.zip`, `pneumonia.zip`) if not already extracted.
2. Open [`test.ipynb`](test.ipynb) or [`test_densenet121.ipynb`](test_densenet121.ipynb) in VS Code or Jupyter.
3. Run all cells to start federated training and evaluation.
4. Check the `metrics_log.csv` and `federated_metrics_plots/` for results and plots.

## Citation

If you use the COVID-19 Radiography Dataset, please cite the sources listed in [`covid19/COVID-19_Radiography_Dataset/README.md.txt`](covid19/COVID-19_Radiography_Dataset/README.md.txt).

---

**Author:** Bikram Mukherjee  
