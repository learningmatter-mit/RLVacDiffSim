#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 13:18:49 2024

@author: ubuntu
"""

import json

import numpy as np
import torch
from ase import io
from rgnn.common.registry import registry
from rgnn.graph.atoms import AtomsGraph
from rgnn.graph.utils import batch_to

from rlsim.utils.sro import get_sro

device = 'cuda:0'
vacancy = 1
ideal_n_atoms = 256
defect = vacancy/ideal_n_atoms
q_params = {"temperature": 300};
T = q_params['temperature'];
# tau = 30
model = registry.get_model_class("t_net").load("dev/t_model_combined_0724_30.pth.tar")
model.eval()
model.to(device)


def WC_SRO(atoms):

    species = list(set(atoms.get_atomic_numbers().tolist()));
    n_species = len(species);
    n_atoms = len(atoms);
    dist_all = atoms.get_all_distances(mic=True);
    dist_list = dist_all.flatten();
    dist_list = dist_list[dist_list>0.1];
    NN_dist = np.min(dist_list);
    r_cut = (1+np.sqrt(2))/2*NN_dist;

    pairs = (dist_all>0.1)*(dist_all<r_cut);
    total_pairs = np.sum(pairs)/2;
    atomic_numbers = atoms.get_atomic_numbers();
    specie_list = [atomic_numbers==species[n] for n in range(n_species)];
    n_specie_list = [np.sum(specie_list[n]) for n in range(n_species)];
    n_specie_list = np.array(n_specie_list)/np.sum(n_specie_list);
    SRO = np.zeros((n_species,n_species));
    
    for n1 in range(n_species):
        for n2 in range(n1+1):
            
            number_of_pairs = np.sum(pairs*specie_list[n1][None,:]*specie_list[n2][:,None])/2;
            SRO[n1,n2] = 1 - (number_of_pairs/total_pairs)/n_specie_list[n1]/n_specie_list[n2];
            
            if(n1 != n2):
                SRO[n2,n1] = SRO[n1,n2];
       
    return SRO;

## Original

def timer_converter(t, tau=30, threshold=0.9999):
    if t < threshold * tau:
        return -tau * torch.log(1 - t / tau)
    else:
        # Linear approximation for large t values
        return tau * (t / tau - 1 + torch.log(torch.tensor(threshold, device=t.device, dtype=t.dtype)))
    
atoms_l = [io.read('dev/DQN_'+ str(ideal_n_atoms-vacancy) +'_'+ str(T) + 'K/XDATCAR'+str(k), index=':') for k in range(10)];
print('dev/paper/time/time_map_'+str(T)+'K_'+str(ideal_n_atoms)+'_'+str(vacancy)+ '_'+str(model.tau)+'_combined_0724.json')
out = [];
sro_out = []
embedding = []

for j in range(int(len(atoms_l[:][0])/10)):
    i = 10*j;
    out.append([]);
    sro_out.append([])
    for k in range(10):
        data = AtomsGraph.from_ase(atoms_l[k][i],
                                model.cutoff,
                                read_properties=False,
                                neighborlist_backend="ase",
                                add_batch=True)
        batch = batch_to(data, device)
        pred_time = model(batch, temperature = T, defect=defect)
        # time = model.get_time_from_goal_state(pred)
        # time_real = -model.tau*torch.log(1-pred_time/model.tau)
        time_real = timer_converter(pred_time, model.tau)
        # print(len(atoms_l[k][i]), atoms_l[k][i].get_pbc(),pred_time.item(), time_real.item())
        # out[-1] += [float(time.detach())];
        out[-1] += [float(time_real.detach())];
        SRO = get_sro(atoms_l[k][i])
        sro_norm = np.linalg.norm(SRO)
        sro_out[-1] += [sro_norm]
    if(i%20==0):
        print('complete '+str(i//20+1)+'%');
        
with open('dev/paper/time/time_map_'+str(T)+'K_'+str(ideal_n_atoms)+'_'+str(vacancy)+ '_'+str(model.tau)+'_combined_0724.json','w') as file:
    json.dump(out,file);
with open('dev/paper/time/sro_map_'+str(T)+'K_'+str(ideal_n_atoms)+'_' +str(vacancy)+ '_combined_0724.json','w') as file:
    json.dump(sro_out,file);


# dataset_path = "/home2/hojechun/00-research/14-time_estimation/dataset_900.json"
# with open(dataset_path) as file:
#     data_read = json.load(file);

# data = data_read

# atoms_list = [Atoms(positions = state['state']['positions'],
#                 cell = state['state']['cell'],
#                 numbers = state['state']['atomic_numbers'],
#                 pbc=[1,1,1]) for state in data];
# random_idx = np.random.choice(len(atoms_list), size=900, replace=False)
# out = [];
# for k in random_idx:
#     data = AtomsGraph.from_ase(atoms_list[k],
#                             model.cutoff,
#                             read_properties=False,
#                             neighborlist_backend="ase",
#                             add_batch=True)
#     batch = batch_to(data, device)
#     pred_time = model(batch, temperature = T, defect=defect)
#     # print(pred)
#     time_real = -model.tau*torch.log(1-pred_time/model.tau)
#     print(len(atoms_list[k]), atoms_list[k].get_pbc(),pred_time.item(), time_real.item())
#     out.append(float(time_real.detach()))
   
# with open('time_map_dataset_'+str(T)+'_' +str(vacancy)+'_'+str(model.tau)+'.json','w') as file:
#     json.dump(out,file);
