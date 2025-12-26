# Hybrid Network for Upper Gastrointestinal Disease Segmentation with Illumination–Boundary Adaptation and Uncertainty-Aware Post-Hoc Refinement

This repository provides the official codes of DySSNet with MSM-TTA and the UGIAD-Seg dataset. This repository is under active development. Inference code and examples coming soon!

Enhanced disease segmentation model with Retinex illumination module and Swin-UMamba backbone.


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


<details> <summary>Click to expand manual installation steps</summary>
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

</details> ```

