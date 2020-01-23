import torch
from force_attract_with_dest import force_goal, pose_propagation
import numpy as np
import math
# param.ped_radius, param.ped_mass, param.betta

class Repulsive_forces():
    def __init__(self):
        self.num_ped = None
        self.aux1 = None
        self.auxullary = None
        self.aux = None
        self.indexes        = None
        self.uneven_indexes = None
        self.even_indexes   = None
    
    def change_num_of_ped(self, new_num):
        self.num_ped = new_num
        self.aux1 = None
        self.auxullary = None
        self.aux = None
        self.indexes        = None
        self.uneven_indexes = None
        self.even_indexes   = None
        self.generate_aux_matrices()

    def generate_aux_matrices(self):
        if self.aux1 is None:
            self.aux1 = torch.tensor(([1., 0.], [0., 1.]))
            for i in range(0, self.num_ped):
                self.aux1 = torch.cat((self.aux1, torch.tensor(([1., 0.], [0., 1.]))), dim=1)
            # self.aux1.requires_grad_(True)
            # self.aux1.retain_grad()
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
        if self.auxullary is None:
            self.auxullary = torch.zeros(self.num_ped+1, (self.num_ped+1)*2)
            for i in range(self.num_ped+1):
                self.auxullary[i, 2*i] = 1.
                self.auxullary[i, 2*i+1] = 1.

                ''' used to calc x+y of each agent pose
                    auxullary  tensor([
                        [1., 1., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 1., 1., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 1., 1., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 1., 1.]])
                '''
        if self.aux is None:
            self.aux = self.auxullary.t()
            # self.aux.retain_grad()
        
        if self.indexes is None:
            self.indexes = np.linspace(0., (self.num_ped+1)*2, ((self.num_ped+1)*2+1))
            self.uneven_indexes = self.indexes[1:-1:2]
            self.even_indexes = self.indexes[0:-1:2]


    def calc_rep_forces(self, state, A=10, ped_radius=0.3, ped_mass=60, betta=0.08, velocity_state = None, param_lambda = 1):
        if (state.shape[0]-1) != self.num_ped:
            self.change_num_of_ped(state.shape[0]-1)

        state_concated = state.clone().matmul(self.aux1)
        # state_concated.retain_grad()
        state_concated_t = state.reshape(1, -1)
        for i in range(0, state.shape[0]-1):
            state_concated_t = torch.cat([state_concated_t, state.reshape(1, -1)])
            '''    state_concated_t tensor(
            [[ 1.,  0.,  2.,  1., -1., -1.,  0., -1.],
            [ 1.,  0.,  2.,  1., -1., -1.,  0., -1.],
            [ 1.,  0.,  2.,  1., -1., -1.,  0., -1.],
            [ 1.,  0.,  2.,  1., -1., -1.,  0., -1.]]
            '''
        delta_pose = (-state_concated_t + state_concated)
        delta_pose += 0.0000001
        
        dist_squared = ((delta_pose)**2)
        # dist_squared.retain_grad()
        # used to calc delta_x**2 +delta_y**2 of each agent
        
        # sqrt(delta_x**2 +delta_y**2) -> distance
        # TODO: otherwise  when doing backprop: sqrt(0)' -> nan
        # dist_squared += 0.0000001
        dist = (dist_squared.matmul(self.aux))
        # aka distance
        dist = torch.sqrt(dist) + 10000000 * \
            torch.eye(dist.shape[0])  # TODO: deal with 1/0,

        # formula(21) from `When Helbing Meets Laumond: The Headed Social Force Model`
        # according to Headed Social Force Model
        
        force_amplitude = A * torch.exp((ped_radius - dist) / betta)
        
        # print ("delta_pose / (dist).matmul(auxullary) \n",delta_pose / (dist).matmul(auxullary))
        # formula(21) from `When Helbing Meets Laumond: The Headed Social Force Model`

        velocity_state_concated = velocity_state.clone().matmul(self.aux1)
        
        velocity_atan = torch.atan2(velocity_state_concated[:,self.uneven_indexes], velocity_state_concated[:,self.even_indexes])
        # print ("delta_pose", delta_pose)
        dy = delta_pose[:,self.uneven_indexes]
        dx = delta_pose[:,self.even_indexes]
        deltapose_atan = torch.atan2(-dy, -dx)
        phi = ((velocity_atan - deltapose_atan) + math.pi) % (2*math.pi) - math.pi
        
        
        anisotropy = param_lambda + (1 - param_lambda)*(1+torch.cos(phi))/2.
        anisotropy = anisotropy.matmul(self.auxullary)
        force = force_amplitude.matmul(
            self.auxullary)*(delta_pose / (dist).matmul(self.auxullary)) * anisotropy
        # print ("force BBbefore: ", force)
        force = (force * ((self.auxullary - 1) * -1))
        # print ("force before: ", force)
        aux2 = self.aux1.clone().t()
        force = force.matmul(aux2)
        # print ("force after: ", force)
        # print ()
        return force


rep_f = Repulsive_forces()
def calc_forces(state, goals, pedestrians_speed, robot_speed, k, alpha, ped_radius, ped_mass, betta, param_lambda = 1):

    rep_force = rep_f.calc_rep_forces(state[:, 0:2], alpha, ped_radius, ped_mass, betta, state[:,2:4], param_lambda)
    attr_force = force_goal(state, goals, pedestrians_speed,robot_speed, k)
    return rep_force, attr_force

# def calc_forces_(state, goals, pedestrians_speed, k, alpha, ped_radius, ped_mass, betta, param_lambda = 1):
#     rep_force = calc_rep_forces(
#         state[:, 0:2], alpha, ped_radius, ped_mass, betta, state[:,2:4], param_lambda)
#     attr_force = force_goal(state, goals, pedestrians_speed, k)
#     return rep_force, attr_force


def calc_cost_function(a, b, e, goal, init_pose, agents_pose, policy=None):
    robot_pose = agents_pose[0, 0:2].clone()
    robot_speed = agents_pose[0, 2:4].clone()
    PG = (robot_pose - init_pose).dot((-init_pose +
                                      goal[0])/torch.norm(-init_pose+goal[0]))  
    # Blame
    B = torch.zeros(len(agents_pose), requires_grad=False)
    # if torch.norm(robot_speed) > e:
    agents_speed = agents_pose[:,0:2]
    delta =  agents_speed - robot_pose
    norm = -torch.norm(delta, dim = 1)/b
    B = torch.exp(norm)
    # Overall Cost
    B = (-a*PG+1*B) #*k_dist[n]
    # if policy != None:
    #     print(policy+ " PG ",PG)
    #     print("B  ",B)

    return B + 1000


def calc_rep_forces(state, A=10, ped_radius=0.3, ped_mass=60, betta=0.08, velocity_state = None, param_lambda = 1):
    # state = state_[:,0:2]

    # used to transform state from [N_rows*2] to [N_rows*(2*N_rows)]
    aux1 = torch.tensor(([1., 0.], [0., 1.]))
    for i in range(0, state.shape[0]-1):
        aux1 = torch.cat((aux1, torch.tensor(([1., 0.], [0., 1.]))), dim=1)
    aux1.requires_grad_(True)
    aux1.retain_grad()
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
    state_concated = state.clone().matmul(aux1)
    state_concated.retain_grad()
    state_concated_t = state.reshape(1, -1)
    for i in range(0, state.shape[0]-1):
        state_concated_t = torch.cat([state_concated_t, state.reshape(1, -1)])
        '''    state_concated_t tensor(
        [[ 1.,  0.,  2.,  1., -1., -1.,  0., -1.],
         [ 1.,  0.,  2.,  1., -1., -1.,  0., -1.],
         [ 1.,  0.,  2.,  1., -1., -1.,  0., -1.],
         [ 1.,  0.,  2.,  1., -1., -1.,  0., -1.]]
        '''
    delta_pose = (-state_concated_t + state_concated)
    delta_pose += 0.0000001
    auxullary = torch.zeros(state_concated.shape[0], state_concated.shape[1])
    for i in range(state_concated.shape[0]):
        auxullary[i, 2*i] = 1.
        auxullary[i, 2*i+1] = 1.

    ''' used to calc x+y of each agent pose
        auxullary  tensor([
            [1., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 1.]])
    '''
    dist_squared = ((delta_pose)**2)
    dist_squared.retain_grad()
    # used to calc delta_x**2 +delta_y**2 of each agent
    aux = auxullary.t()
    aux.retain_grad()
    # sqrt(delta_x**2 +delta_y**2) -> distance
    # TODO: otherwise  when doing backprop: sqrt(0)' -> nan
    # dist_squared += 0.0000001
    dist = (dist_squared.matmul(aux))
    # aka distance
    dist = torch.sqrt(dist) + 10000000 * \
        torch.eye(dist.shape[0])  # TODO: deal with 1/0,

    # formula(21) from `When Helbing Meets Laumond: The Headed Social Force Model`
    # according to Headed Social Force Model
    
    force_amplitude = A * torch.exp((ped_radius - dist) / betta)
    
    # print ("delta_pose / (dist).matmul(auxullary) \n",delta_pose / (dist).matmul(auxullary))
    # formula(21) from `When Helbing Meets Laumond: The Headed Social Force Model`

    velocity_state_concated = velocity_state.clone().matmul(aux1)
    indexes = np.linspace(0., velocity_state_concated.shape[1], (velocity_state_concated.shape[1]+1))
    uneven_indexes = indexes[1:-1:2]        
    even_indexes = indexes[0:-1:2]
    velocity_atan = torch.atan2(velocity_state_concated[:,uneven_indexes], velocity_state_concated[:,even_indexes])
    # print ("delta_pose", delta_pose)
    dy = delta_pose[:,uneven_indexes].clone()
    dx = delta_pose[:,even_indexes].clone()
    deltapose_atan = torch.atan2(-dy, -dx)
    phi = ((velocity_atan - deltapose_atan) + math.pi) % (2*math.pi) - math.pi
    
    
    anisotropy = param_lambda + (1 - param_lambda)*(1+torch.cos(phi))/2.
    anisotropy = anisotropy.matmul(auxullary)
    force = force_amplitude.matmul(
        auxullary)*(delta_pose / (dist).matmul(auxullary)) * anisotropy
    # print ("force BBbefore: ", force)
    force = (force * ((auxullary - 1) * -1))
    # print ("force before: ", force)
    aux2 = aux1.clone().t()
    force = force.matmul(aux2)
    # print ("force after: ", force)
    # print ()
    return force
