import sys
print(sys.version)
import torch
import torch.nn as nn

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import OrderedDict

# Import PyTorch

from torch.autograd import Variable

from torch.autograd import Function # import Function to create custom activations
from torch.nn.parameter import Parameter # import Parameter to create custom activations with learnable parameters
from torch import optim # import optimizers for demonstrations
import torch.nn.functional as F # import torch functions
from torchvision import datasets, transforms # import transformations to use for demo


def tensor_inf_to_zero(bad_tensor):
    bad_tensor[bad_tensor == float('inf')] = 0
    bad_tensor.requires_grad_(True)
    bad_tensor.retain_grad()
    return bad_tensor


def calc_forces_1(state):

    aux1 = torch.tensor(([1.,0.],[0.,1.])) # used to transform state from [N_rows*2] to [N_rows*(2*N_rows)]
    '''
        e.g. : state   [[1,  0]
                        [2,  1]
                        [-1,-1]
                        [0, -1]]
        
        new state_concated:         
                       [[1,  0, 1,  0, 1,  0, 1,  0]
                        [2,  1, 2,  1, 2,  1, 2,  1]
                        [-1,-1, -1,-1, -1,-1, -1,-1]
                        [0, -1, 0, -1, 0, -1, 0, -1]]
    '''

    for i in range(0,state.shape[0]-1):
        aux1 = torch.cat((aux1,torch.tensor(([1.,0.],[0.,1.]))),dim=1)
    
    
    aux1.requires_grad_(True)
    aux1.retain_grad()
    
    state_concated = state.matmul(aux1).requires_grad_(True)
    state_concated.retain_grad()
    
    
    state_concated_t = state.view(1,-1).requires_grad_(True)
    for i in range(0,state.shape[0]-1):
        state_concated_t = torch.cat([state_concated_t,state.view(1,-1)]).requires_grad_(True)
        
        '''    state_concated_t tensor(
        [[ 1.,  0.,  2.,  1., -1., -1.,  0., -1.],
         [ 1.,  0.,  2.,  1., -1., -1.,  0., -1.],
         [ 1.,  0.,  2.,  1., -1., -1.,  0., -1.],
         [ 1.,  0.,  2.,  1., -1., -1.,  0., -1.]]
        '''
    
    state_concated_t.requires_grad_(True)

    delta_pose = (state_concated_t - state_concated).requires_grad_(True)
    delta_pose.retain_grad()
    
    auxullary = torch.zeros(state_concated.shape[0], state_concated.shape[1])
    auxullary.retain_grad()
    for i in range(state_concated.shape[0]):

        auxullary[i,2*i] = 1.
        auxullary[i,2*i+1] = 1.

    ''' used to calc x+y of each agent pose
        auxullary  tensor([
            [1., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 1.]])
    '''

    dist_squared = ((delta_pose)**2).requires_grad_(True)
    dist_squared.retain_grad()
    
    # used to calc delta_x**2 +delta_y**2 of each agent
    aux = auxullary.t()
    aux.retain_grad()
    
    # sqrt(delta_x**2 +delta_y**2) -> distance
    dist_squared += 0.0000001 # TODO: otherwise  when doing backprop: sqrt(0)' -> nan
    print ("aux ",aux)
    print ("dist_squared ",dist_squared)
    dist = torch.sqrt(dist_squared.matmul(aux)).requires_grad_(True) ## aka distance
    print ("dist ",dist)
    A = 2 * (10**3) # const param from  formula(21) from `When Helbing Meets Laumond: The Headed Social Force Model`
    force_amplitude = A * torch.exp((0.3 -dist)/0.08).requires_grad_(True) ## according to Headed Social Force Model
    

    force = force_amplitude.matmul(auxullary) * delta_pose/(dist+0.00001).matmul(auxullary) # formula(21) from `When Helbing Meets Laumond: The Headed Social Force Model`
    
    # force[force != force] = 0 deal with nans
    force.requires_grad_(True)
    force.retain_grad()    

    aux1 = aux1.t()
    force = force.matmul(aux1).requires_grad_(True)
    # print ("force ", force)
    
    # force.register_hook(lambda grad: print(grad)) 
    # aux1.register_hook(lambda grad: print(grad)) 
    # force_amplitude.register_hook(lambda grad: print(grad)) 
    # delta_pose.register_hook(lambda grad: print(grad)) 
    # dist.register_hook(lambda grad: print(grad)) 

    force.backward(state)
    gr = state.grad
    
    return force ,gr

data = torch.randn((4, 2),requires_grad=True)
# print ("\ninput: ",data)
force, gr = calc_forces_1(data)
# print("\n\nforces:\n" ,force)

# print("\ngr:\n" ,gr)



