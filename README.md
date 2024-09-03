# RLVacDiffSim
Reinforcement Learning driven simulation of vacancy diffusion

## TODO

- [ ] Debug script
- [ ] Documentation

## Create conda envrionment

```bash
conda update conda
pip install --upgrade pip
cd <rlsim_direcotry>
conda env create -f environment.yml
conda activate rlsim-env
```

## Install `pytroch`, `torch_geometric` right cuda version

- below example is for pytorch version 2.2.0 with cuda version 12.1

```bash

conda install pytorch=2.2.0 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

```

## Install the package

```bash
pip install -e .
pip install -e ".[dev]" # For developer version
```

## Install `MACE`, `RGNN`
- `MACE`: `pip install mace-torch`
- `RGNN`: install from [github page]("https://github.com/HojeChun/ReactionGraphNeuralNetwork")
