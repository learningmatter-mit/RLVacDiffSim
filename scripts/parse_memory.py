import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ase import io

from rlsim.memory import Memory


def load_data(dataset='training', folder="../data/Vrandom_mace", run_n=5, traj_n=100):
    traj = Memory(1,0)
    for u1 in range(run_n):
        for i in range(traj_n):
            
            if dataset == 'training':
                criterion = (i%3 <= 1)
            elif dataset == 'testing':
                criterion = (i%3 > 1)
            else:
                raise('dataset must be either training or testing')
            print(f'{folder}/traj'+str(u1)+'/traj'+str(i)+'.json')
            if criterion and os.path.exists(f'{folder}/traj'+str(u1)+'/traj'+str(i)+'.json'):
                traj_list = Memory(1,0)
                traj_list.load(f'{folder}/traj'+str(u1)+'/traj'+str(i))
                traj.rewards += traj_list.rewards[:-1]
                traj.states += traj_list.states[:-1]
                traj.next_states += traj_list.next_states[:-1]
                traj.act_space += traj_list.act_space[:-1]
                traj.actions += traj_list.actions[:-1]
                traj.freq += traj_list.freq[:-1]
                traj.E_min += traj_list.E_min[:-1]
                traj.E_next += traj_list.E_next[:-1]

    nframes = len(traj.rewards)
    for u in range(nframes):
        v = nframes - u - 1
        if traj.freq[v] <= 0 or traj.rewards[v] <- 5 or np.abs(traj.E_next[v] - traj.E_min[v]) > 3 or np.isnan(traj.freq[v]) or np.isnan(traj.rewards[v]):
            del traj.freq[v]
            del traj.rewards[v]
            del traj.states[v]
            del traj.next_states[v]
            del traj.act_space[v]
            del traj.actions[v]
            del traj.E_min[v]
            del traj.E_next[v]
    
    return traj

