# DeepCosmoNet Voids
#### A Multi-Scale 3D Deep Learning Approach for Detecting and Segmenting Cosmic Voids in Large-Scale Underdense Cosmic Web Structures.

DeepCosmoNet Voids utilizes a 3D YOLO-like detector architecture featuring super-separable 3D convolutional blocks, multi-anchor support, and a logarithmic sliding window on radius subsets to accurately identify 3D cosmic voids from massive N-body simulation snapshots.

## 🚀 Installation
Ensure you have a Python environment ready (Python 3.10+ recommended). Install the mandatory dependencies using the requirements.txt file:

```
git clone https://github.com/koexai/DeepCosmoNet_voids.git
cd DeepCosmoNet-Voids
pip install -r requirements.txt
```
## 📂 Data Setup
The dataset structure is completely standardized and automatically parsed by src.dcn_config. Dataset is available on our website [deepcosmonet.koexai.com](deepcosmonet.koexai.com)

Before running training, make sure your data directory mirrors the structure below:
```
DeepCosmoNet-Voids/
├── data/
│   ├── voxel_particles_train/   # 3D density distribution grids for training
│   ├── voxel_particles_val/     # 3D density distribution grids for validation
│   ├── voxel_void_train/        # Ground-truth void catalogs/dataframes for training
│   └── voxel_void_val/          # Ground-truth void catalogs/dataframes for validation
```
### Subfolder Setup Instructions
Particles Data (voxel_particles_train/ & voxel_particles_val/): Should contain preprocessed data arrays representing voxelized simulation blocks.

Void Targets (voxel_void_train/ & voxel_void_val/): Must contain corresponding pandas-readable dataframes detailing true cosmic voids with explicit column identifiers for spatial positioning and sizing ('x', 'y', 'z', 'radius').

## 🏗️ Project Structure
DeepCosmoNet-Voids/

```
├── data/                       # Local data directories
├── docs/                       # API documentation Sphinx generated
├── output/                     # Saved models checkpoints and evaluation logs
├── src/                        # Main package source code
│   ├── __init__.py
│   ├── augmentation.py         # 3D random axis flips and permutations
│   ├── dataset.py              # Dataset/DataLoader with multi-anchor logic
│   ├── dcn_config.py           # Global paths management & configuration constants
│   ├── features.py             # Gaussian smoothing at multi-scale Sigmas
│   ├── filter_overlap.py       # Post-processing filters
│   ├── full_grid_tensor.py     # Grid target converters & postprocessing logic
│   ├── log_experiments.py      # Automates code zipping for experiment tracking
│   ├── logger_matplotlib.py    # Training loss plotting & visualizations
│   ├── main.py                 # Full experiment execution
│   └── train.py                # Main training logic
└── requirements.txt            # System dependencies
```

## 🏋️ Training the Model
Training builds multi-channel scale optimizations dynamically using Gaussian smoothing across different radii, leverages real-time 3D data augmentations (flips and coordinate permutations), and organizes targets via KDTree mappings.

To launch the training cycle wrapper locally or on your cluster node:

### Run Training
```
python src/train.py
```

## 🧪 Experiments & Evaluation
The outputs of the training (pre-trained models, logs, plots ...) will be available in the output folder.
Each run will have its own timestamp generated folder with all the necessary to reproduce the experiment.

## References
Paper: [**3D YOLO-like detector for cosmic voids: A multi-scale deep learning approach to large-scale underdense structures** *G.Puglisi et al*](https://www.sciencedirect.com/science/article/pii/S2213133726000946)