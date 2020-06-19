import torch
import numpy as np
import math


class RepulsiveForces():
    def __init__(self, param):
        self.param = param
        self.num_ped = None
        self.aux1 = None
        self.auxullary = None
        self.aux = None
        self.indexes = None
        self.uneven_indexes = None
        self.even_indexes = None
        self.device = None
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.device = torch.device("cpu")

    def change_num_of_ped(self, new_num):
        self.num_ped = new_num
        self.aux1 = None
        self.aux2 = None
        self.auxullary = None
        self.aux = None
        self.indexes = None
        self.uneven_indexes = None
        self.even_indexes = None
        self.generate_aux_matrices()

    def generate_aux_matrices(self):
        if self.aux1 is None:
            self.aux1 = torch.tensor(([1., 0.], [0., 1.]))
            if self.device is not None:
                self.aux1 = self.aux1.to(self.device)
            for i in range(0, self.num_ped):
                temp = torch.tensor(([1., 0.], [0., 1.]))
                if self.device is not None:
                    temp = temp.to(self.device)
                self.aux1 = torch.cat((self.aux1, temp), dim=1)
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
            if self.device is not None:
                self.auxullary = self.auxullary.to(self.device)
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
            if self.device is not None:
                self.aux = self.aux.to(self.device)

        if self.indexes is None:
            self.indexes = np.linspace(
                0., (self.num_ped+1)*2, ((self.num_ped+1)*2+1))
            self.uneven_indexes = self.indexes[1:-1:2]
            self.even_indexes = self.indexes[0:-1:2]

        if self.aux2 is None:
            self.aux2 = self.aux1.clone().t()

    def calc_rep_forces(self, state, velocity_state, param_lambda):
        num_ped = state.shape[0]-1
        if num_ped != self.num_ped:
            self.change_num_of_ped(num_ped)

        pr = self.param.socForcePersonPerson["d"] * \
            torch.ones(num_ped+1, num_ped+1)
        pr[0, :] = self.param.socForceRobotPerson["d"]
        pr[:, 0] = self.param.socForceRobotPerson["d"]
        pm = self.param.ped_mass
        betta = self.param.socForcePersonPerson["B"] * \
            torch.ones(num_ped+1, num_ped+1)
        betta[0, :] = self.param.socForceRobotPerson["B"]
        betta[:, 0] = self.param.socForceRobotPerson["B"]
        alpha = self.param.socForcePersonPerson["A"] * \
            (1 - torch.eye(self.num_ped+1, self.num_ped+1))
        alpha[0, :] = self.param.socForceRobotPerson["A"]
        alpha[:, 0] = self.param.socForceRobotPerson["A"]

        state_concated = state.clone().matmul(self.aux1)
        state_concated_t = state.reshape(1, -1)
        for i in range(0, state.shape[0]-1):
            state_concated_t = torch.cat(
                [state_concated_t, state.reshape(1, -1)])
        '''    state_concated_t tensor(
        [[ 1.,  0.,  2.,  1., -1., -1.,  0., -1.],
        [ 1.,  0.,  2.,  1., -1., -1.,  0., -1.],
        [ 1.,  0.,  2.,  1., -1., -1.,  0., -1.],
        [ 1.,  0.,  2.,  1., -1., -1.,  0., -1.]]
        '''
        delta_pose = (-state_concated_t + state_concated) + 1e-6

        dist_squared = delta_pose ** 2
        dist = (dist_squared.matmul(self.aux))
        # aka distance
        temp = torch.eye(dist.shape[0])
        if self.device is not None:
            temp = temp.to(self.device)
        dist = torch.sqrt(dist) + 10000000 * \
            temp  # TODO: deal with 1/0,
        force_amplitude = alpha * torch.exp((pr - dist) / betta)
        force = force_amplitude.matmul(
            self.auxullary)*(delta_pose / (dist).matmul(self.auxullary)) # * anisotropy

        force = (force * ((self.auxullary - 1) * -1))
        force = force.matmul(self.aux2)
        yaw_zeros = torch.zeros(len(force), 1)
        force = torch.cat((force, yaw_zeros), dim=1)
        return force
