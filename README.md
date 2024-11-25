# RLVacDiffSim
Reinforcement Learning driven simulation of vacancy diffusion

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

## Usage
We provide scripts in command line interface (CLI).
Trained models and initial poscars (256 atoms with mono vacancy) are saved in [zenodo]("TBD")
Exampeles are as follows:

- Generate dataset for pre-trained reaction encodings

```
rlsim-gen_pretrain_data -c '/path/to/config'
```

- Train a model for deep reinforcement learning (DRL)
```
rlsim rl-train -c '/path/to/context_bandit/config' # Contextual Bendit
rlsim rl-train -c '/path/to/dqn/config' # Deep Q Network training
```
- Deploy DRL 
```
rlsim rl-deploy -c '/path/to/tks/config' # Transition kinetics simulation
rlsim rl-deploy -c '/path/to/dqn/config' # Lower-energy state sampling
```
- Generate dataset for time estimator
```
rlsim-gen_time_dataset -f '/path/to/poscars` -c '/path/to/config' -s 30 -n 100
```
- Train a time estimator
```
rlsim time-train -c '/path/to/time/config' 
```
- Estimate time using the time estimator
```
rlsim-estimate_time -m t_net_binary '/path/to/model' -v 1 256 -t 300 -i '/path/to/trajectory -n 10 -s '/path/to/save_dir' -d cuda
```
examples of configurations are saved in `conifgs`

## Citation
```
TBD
```