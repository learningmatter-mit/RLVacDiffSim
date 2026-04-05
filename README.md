# RLVacDiffSim
Reinforcement Learning driven simulation of vacancy diffusion

## System requirements
The package works in Linux systems. The versions of dependent packages the software has been tested on are listed below:

python==3.10.13 numpy=1.26.4 scipy==1.15.3 matplotlib==3.8.2 torch==2.2.0 torch_cluster==1.6.3 torch_scatter==2.1.2 e3nn==0.4.4 
torch_geometric==2.7.0, torch_sparse==0.6.18, torch_spline_conv==1.2.2

cuda version: 12.1

Note that in most cases, different version of packages should also work. We list exactly the versions in our calculations in case version inconsistency issue occurs. If users intend to run the program on a cpu device, the cuda package is not needed.

## Installation guide
### Create conda envrionment

```bash
conda update conda
pip install --upgrade pip
cd <rlsim_direcotry>
conda env create -f environment.yml
conda activate rlsim-env
```

### Install `pytroch`, `torch_geometric` right cuda version

- below example is for pytorch version 2.2.0 with cuda version 12.1

```bash

conda install pytorch=2.2.0 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install torch_geometric

```

### Install the package

```bash
pip install -e .
pip install -e ".[dev]" # For developer version
```

## Demo
Here we provide a demo task to run the Deep RL low-energy states sampling (LSS), calling our pre-trained RL model and the MACE-MP-0 interatomic potential. The task is running on the equiatomic CrCoNi alloy. The following commands are a minimal example to launch the calculation:
```bash
cd demo
wget https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model
rlsim rl-deploy -c config.toml
```
The expected command line output is shown below, expected to be completed in a couple of minutes:
```
INFO:Deploy:Step: 0, T: 1200, E: -1595.165
INFO:Deploy:Step: 10, T: 700, E: -1595.011
INFO:Deploy:Step: 19, T: 700, E: -1595.216
INFO:Deploy:Simulation finished.
```
The output files of simulation trajectory and energies (at energy local minimums at each timestep) are in "demo/test_run_lss/XDATCAR0" and "demo/test_run_lss/converge.json", respectively.

## Instructions for Use
We provide scripts in command line interface (CLI).
Trained models and initial poscars (256 atoms with mono vacancy) are saved in figshare [TBD]

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
@misc{chun2024learningmeanpassagetime,
      title={Learning Mean First Passage Time: Chemical Short-Range Order and Kinetics of Diffusive Relaxation}, 
      author={Hoje Chun and Hao Tang and Rafael Gomez-Bombarelli and Ju Li},
      year={2024},
      eprint={2411.17839},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2411.17839}, 
}
```
