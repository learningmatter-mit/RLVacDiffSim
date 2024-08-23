from heapq import nsmallest
from itertools import product

import ase
import numpy as np
from numpy.linalg import norm
from scipy.spatial import ConvexHull


def get_action_space(config):
    
    a = 3.528;
    pos = config.atoms.get_positions();
    L = config.atoms.cell[0,0];
    n_atom = len(pos);
    actions = [];
    
    dist_mat = config.atoms.get_all_distances(mic=True);

    crit = np.sum(dist_mat<a/np.sqrt(2)*1.2, axis=1);
    vacancy_l = np.argwhere(crit != 13).T[0];
    
    def mic(yn):
        
        dr = (yn +L/2)%L - L/2;
        
        return dr;
    
    def test(i, vec):
        
        test = config.atoms.copy();
        pos_test = test.get_positions();
        pos_test[i] += vec;
        test.set_positions(pos_test);
        
        return np.sum(test.get_distances(i, range(len(test)), mic=True) < 0.8) == 1;
    
    acts = np.array([[1,1,0],
                     [1,-1,0],
                     [-1,1,0],
                     [-1,-1,0],
                     [1,0,1],
                     [1,0,-1],
                     [-1,0,1],
                     [-1,0,-1],
                     [0,1,1],
                     [0,1,-1],
                     [0,-1,1],
                     [0,-1,-1]])*a/2*0.8;
                    
    for i in vacancy_l:
        
        for vec in acts:
            vacant = test(i, vec);
            if(vacant):
                actions.append([i]+vec.tolist());
        
    return actions;

