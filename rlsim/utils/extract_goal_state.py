#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 13:18:49 2024

@author: ubuntu
"""

import json
import sys

import numpy as np
import torch
from ase import Atoms, io
from rgnn.common.registry import registry
from rgnn.graph.atoms import AtomsGraph
from rgnn.graph.utils import batch_to
from torch.nn import MSELoss
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader

from .sro import get_sro

vacancy = 1
T = 500

## Original

atoms_l = [io.read('dev/paper/DQN_'+ str(256-vacancy) +'_'+ str(T) + 'K/XDATCAR'+str(k), index=':') for k in range(10)];

out = [];
sro_out = []
for j in range(190):
    i = 10*j;
    out.append([]);
    sro_out.append([])
    for k in range(10):
        SRO = get_sro(atoms_l[k][i])
        sro_norm = np.linalg.norm(SRO)
        sro_out[-1] += [sro_norm]
    if(i%20==0):
        print('complete '+str(i//20+1)+'%');
        
with open('sro_map_'+str(T)+'_' +str(vacancy)+'_2'+ '.json','w') as file:
    json.dump(sro_out,file);

