# Hybrid Network for Upper Gastrointestinal Disease Segmentation with Illumination–Boundary Adaptation and Uncertainty-Aware Post-Hoc Refinement

This repository provides the official codes of DySSNet with MSM-TTA and the UGIAD-Seg dataset. This repository is under active development. Inference code and examples coming soon!


## 📦 Installation

### Quick Install

```bash
# Clone repository
git clone https://github.com/yourusername/DySSNet.git
cd DySSNet

# Create conda environment
conda create -n dyssnet python=3.10
conda activate dyssnet

# Run automated installation
bash scripts/install.sh
```

### Manual Install
```
# Step 1: Create environment
conda create -n dyssnet python=3.10
conda activate dyssnet

# Step 2: Install PyTorch (Swin-UMamba compatible version)
pip install torch==2.0.1 torchvision==0.15.2

# Step 3: Install Mamba dependencies
pip install causal-conv1d==1.1.1
pip install mamba-ssm

# Step 4: Install other requirements
pip install -r requirements.txt

```

## Dataset Details
The UGIAD-Seg dataset provides open access to 3313 UGI endoscopic images from Macao Kiang Wu Hospital and Xiangyang Centre Hospital, mainly captured using WLE and partly by NBI. These images encompass three key areas: esophagus, stomach, and duodenum, each annotated with specific anatomical landmarks and disease types, and these annotations are both applied and subsequently verified by medical specialists from the two contributing hospitals. The dataset is developed ensuring patient anonymity and privacy, with all materials fully anonymized by excluding patient information from the images and renaming the files according to their anatomical landmark and disease labels, and thereby exempting it from patient consent requirements. The images consist of different resolutions that range between 268x217 and 1545x1156 with most of the black borders removed. 
The dataset can also be downloaded using the following links: <br />
Google Drive: https://drive.google.com/file/d/1TioBa5SoGJF6noxPrqi0iKQkauhiIss6/view?usp=sharing <br />



