# XRobo

### Installation

1. Clone this repository and create an environment.
```bash
git clone git@github.com:Dstate/XRobo.git
conda create -n xrobo python=3.10
conda activate xrobo
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

2. Install flash-attn module.
Based on the current environment, download the corresponding version from [FlashAttention](https://github.com/Dao-AILab/flash-attention/releases). After that, install the `.whl` file using pip.
```bash
pip install ./flash_attn-2.7.1.post4+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

3. Install other package.
```bash
pip install -r requirements.txt
```

### Dataset Preprocessing
1. Prepare datasets.
We save each trajectory as a HDF5 file. You can link your dataset to the repo directory:

```bash
mkdir -p assets/data
ln -s path/to/datasets assets/data/
```

2. Generate meta information of datasets.
Before training, you should generate meta information first, such as `action_max`, `action_min`, `proprio_max`, etc. An example is as follows. `statistics_calc.py` will generate a meta file in the dataset path.
```bash
python statistics_calc.py --dataset_root assets/data/libero
```

### Example for Training and Inference

```bash 
bash scripts/bc_ddpm-libero.sh # Training
python server_vla.py # Inference
```

### Tools

```bash
python scripts/test_dataloader.py # check the structure of dataloader
python scripts/test_client.py # check the vla server
```
