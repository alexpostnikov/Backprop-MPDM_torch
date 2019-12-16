from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# import transformations to use for demo
from torchvision import datasets, transforms
import torch.nn.functional as F  # import torch functions
from torch import optim  # import optimizers for demonstrations
# import Parameter to create custom activations with learnable parameters
from torch.nn.parameter import Parameter
# import Function to create custom activations
from torch.autograd import Function
from torch.autograd import Variable
from collections import OrderedDict
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np  # linear algebra
import time
import torch.nn as nn
import torch
import sys
print(sys.version)


# Import PyTorch


DT = 0.2


def tensor_inf_to_zero(bad_tensor):
    bad_tensor[bad_tensor == float('inf')] = 0
    bad_tensor.requires_grad_(True)
    bad_tensor.retain_grad()
    return bad_tensor


def calc_forces(state):
    rep_force = 10 * calc_rep_forces(state)
    # print ("\n    ---rep_force\n ", rep_force)
    # print ("\n    ---rep_force\n",rep_force)
    attr_force = calc_attractive_forces(state)
    # print ("\n     ---attr_force\n", attr_force)
    # print ("\n    ---attr_force\n",attr_force)
    return rep_force + attr_force


def calc_rep_forces(state):
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

    state_concated = state.matmul(aux1).requires_grad_(True)
    state_concated.retain_grad()

    # print ("state ", state)
    state_concated_t = state.reshape(1, -1).requires_grad_(True)
    for i in range(0, state.shape[0]-1):
        state_concated_t = torch.cat(
            [state_concated_t, state.reshape(1, -1)]).requires_grad_(True)

        '''    state_concated_t tensor(
        [[ 1.,  0.,  2.,  1., -1., -1.,  0., -1.],
         [ 1.,  0.,  2.,  1., -1., -1.,  0., -1.],
         [ 1.,  0.,  2.,  1., -1., -1.,  0., -1.],
         [ 1.,  0.,  2.,  1., -1., -1.,  0., -1.]]
        '''

    state_concated_t.requires_grad_(True)
    # print ("state_concated_t ", state_concated_t)
    # print ("state_concated ", state_concated)
    delta_pose = (state_concated_t - state_concated).requires_grad_(True)

    delta_pose.retain_grad()

    auxullary = torch.zeros(state_concated.shape[0], state_concated.shape[1])
    auxullary.retain_grad()
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

    dist_squared = ((delta_pose)**2).requires_grad_(True)
    dist_squared.retain_grad()

    # used to calc delta_x**2 +delta_y**2 of each agent
    aux = auxullary.t()
    aux.retain_grad()

    # sqrt(delta_x**2 +delta_y**2) -> distance
    # TODO: otherwise  when doing backprop: sqrt(0)' -> nan
    dist_squared += 0.0000001

    dist = (dist_squared.matmul(aux)).requires_grad_(True)  # aka distance

    dist = torch.sqrt(dist) + 10000 * \
        torch.eye(dist.shape[0])  # TODO: deal with 1/0,

    # const param from  formula(21) from `When Helbing Meets Laumond: The Headed Social Force Model`
    A = 2 * (10**3)
    # according to Headed Social Force Model
    force_amplitude = A * torch.exp((0.3 - dist)/0.08).requires_grad_(True)

    # formula(21) from `When Helbing Meets Laumond: The Headed Social Force Model`
    force = force_amplitude.matmul(delta_pose/(dist+0.00001).matmul(auxullary))

    force.requires_grad_(True)
    force.retain_grad()

    aux1 = aux1.t()
    force = force.matmul(aux1).requires_grad_(True)
    # force.backward(state)
    # gr = state.grad

    return force


class LinearFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input):  # , weight, bias=None):

        input, cost = input
        input.requires_grad_(True)
        cost.requires_grad_(True)
        forces = calc_forces(input)
        # print (cost)
        cost += calc_cost_function(agents_pose=input)
        # forces.backward(input)
        # gr = input.grad
        ctx.save_for_backward(input, cost, forces)
        # if bias is not None:
        #     output += bias.unsqueeze(0).expand_as(output)
        return (forces, cost)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        print("here?")
        input, cost, forces = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias


class Linear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(Linear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.

        # self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        # if bias:
        #     self.bias = nn.Parameter(torch.Tensor(output_features))
        # else:
        #     # You should always register all possible parameters, but the
        #     # optional ones can be None if you want.
        #     self.register_parameter('bias', None)

        # Not a very smart way to initialize weights
        # self.weight.data.uniform_(-0.1, 0.1)
        # if bias is not None:
        # self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        input, cost = input
        input.requires_grad_(True)
        cost.requires_grad_(True)
        forces, cost = LinearFunction.apply(
            (input[:, 0:2], cost))  # , self.weight, self.bias)
        input = calc_new_pose(input)
        input = calc_new_vel(input, forces)
        input.requires_grad_(True)
        cost.requires_grad_(True)
        return (input, cost)  # , self.weight, self.bias)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


##########################################################################################
###################################global?################################################
a = 5
b = 2
e = 0.001
robot_speed = 0.1
robot_pose = torch.tensor([1., 2.01], requires_grad=True)
goal = torch.tensor([10., 20.], requires_grad=True)
init_pose = torch.tensor([0., 1.], requires_grad=True)
agents_pose = torch.tensor([[1., 3.], [5., 2.], [1., 0.]], requires_grad=True)

robot_pose.retain_grad()
goal.retain_grad()
init_pose.retain_grad()
agents_pose.retain_grad()


##########################################################################################
##########################################################################################


def calc_attractive_forces(input):
    global goals
    delta_pose = goals - input
    # print ("delta_pose ", delta_pose)
    dist = torch.sqrt(delta_pose[:, 0:1]**2 + delta_pose[:, 1:2]**2)
    force = delta_pose/torch.cat((dist, dist), dim=1)
    return force


def calc_cost_function(a=a, b=b, e=e, robot_speed=robot_speed, robot_pose=robot_pose, goal=goal, init_pose=init_pose, agents_pose=agents_pose):
    costs = torch.zeros(agents_pose.shape, requires_grad=False)
    costs.retain_grad()
    PG = (robot_pose - init_pose)*(-init_pose+goal)/torch.norm(-init_pose+goal)
    PG.retain_grad()
    # Blame
    B = torch.zeros(len(agents_pose), requires_grad=False)
    B.retain_grad()
    if robot_speed > e:
        for n in range(len(agents_pose)):
            # TODO: go into matrix math
            B[n] = torch.exp(-torch.norm(agents_pose[n]-robot_pose)/b)
    # Cost
    for n in range(len(B)):
        costs[n] = -a*PG+B[n]
    return costs


def limit_speed(input_state, limit):

    ampl = torch.sqrt(input_state[:, 0:1]**2 + input_state[:, 1:2]**2)
    mask = torch.cat((ampl > limit, ampl > limit), dim=1)
    ampl_2D = torch.cat((ampl, ampl), dim=1)
    # print (mask)
    # print (input_state[mask])
    # print (ampl_2D[ampl_2D>3])
    input_state[mask] = input_state[mask] * limit / ampl_2D[mask]
    return input_state


def calc_new_vel(input_state, forces):
    # input_vel = input_state[:,2:4].view(-1,2)
    input_state[:, 2:4] = (forces * DT / 6.0) + input_state[:, 2:4]
    # limit max speed :
    input_state[:, 2:3]**2 + input_state[:, 3:4]**2
    input_state[:, 2:4] = limit_speed(input_state[:, 2:4], 0.2)
    return input_state


def calc_new_pose(input):
    input[:, 0:2] = input[:, 0:2] + input[:, 2:4] * DT
    return input


# data = torch.tensor(([2.01,2.1],[2.,2.]),requires_grad=True)
# for i in range(0, 100):
na = numb_agent = 4
# data = torch.rand((na,4),requires_grad=True)
# data[:,2:4] = torch.zeros(na,2)
# goals = 10 * torch.rand((na,2),requires_grad=True)
# data.requires_grad_(True)

data = torch.tensor(([0.1, 2, 0., 0], [-5, -5, 0., 0],
                     [-5, 0, 0., 0], [0, -5, 0., 0]), requires_grad=False)
goals = torch.tensor(([-5, -5.], [-0, -0], [-0, -5],
                      [-5, -0]), requires_grad=True)
# print ("\n  ---input: \n",data)
# print ("\n  ---goals: \n",goals)

# # poses = data[:,0:2]
# # print (poses)
# # poses.requires_grad_(True)
# # cost = calc_cost_function(agents_pose=poses)
# # cost.backward(poses)
# # print (poses.grad)
# # data = torch.rand((na,2),requires_grad=True)
# # cost = calc_cost_function(agents_pose=data)
# # cost.backward(data)
# #
# #

# model = nn.Sequential(OrderedDict([
#           ('fc1', Linear(na,4)),
#           ('fc2', Linear(na,4)),
#           ('fc3', Linear(na,4)),
#           ('fc4', Linear(na,4)),
#           ('fc5', Linear(na,4)),
#           ('fc6', Linear(na,4)),
#           ('fc7', Linear(na,4)),
#           ('fc8', Linear(na,4)),
#           ('fc9', Linear(na,4)),
#           ('fc10', Linear(na,4)),
#           ('fc11', Linear(na,4)),
#           ('fc12', Linear(na,4)),
#           ('fc13', Linear(na,4)),
#           ('fc14', Linear(na,4)),
#           ('fc15', Linear(na,4)),
#           ('fc16', Linear(na,4)),
#           ('fc17', Linear(na,4)),
#           ('fc18', Linear(na,4)),
#           ('fc19', Linear(na,4)),
#           ('fc20', Linear(na,4)),
#           ('fc21', Linear(na,4)),
#           ('fc22', Linear(na,4)),
#           ('fc23', Linear(na,4)),
#           ('fc24', Linear(na,4)),
#           ('fc25', Linear(na,4)),
#           ('fc26', Linear(na,4)),
#           ('fc27', Linear(na,4)),
#           ('fc28', Linear(na,4)),
#           ('fc29', Linear(na,4)),
#         ]))

cost = torch.zeros(na, 2)
# cost.requires_grad_(True)
# data.requires_grad_(True)

l = Linear(na, 4)
# state, cost = l((data, cost))

# # state, cost = model((data, cost))
# # loss =
# # print (state)

# print ("\n    ---cost\n", cost)
# cost.backward(data[:,0:2])
# print ("\n    ---grad:\n", data[:,0:2].grad)


# exit()
x = []
y = []
z = []
x2 = []
y2 = []
x3 = []
y3 = []
x4 = []
y4 = []
t = 0
for i in range(0, 5000):

    # forces, _ = l((data,cost))
    forces = calc_forces(data[:, 0:2])
    data = calc_new_pose(data)
    # print (data.shape)
    # print (forces.shape)

    data = calc_new_vel(data, forces)
    x.append(data.data[0, 0].item())
    y.append(data.data[0, 1].item())
    x2.append(data.data[1, 0].item())
    y2.append(data.data[1, 1].item())
    x3.append(data.data[2, 0].item())
    y3.append(data.data[2, 1].item())
    x4.append(data.data[3, 0].item())
    y4.append(data.data[3, 1].item())
    z.append(t)
    t += DT
    # if i % 10 ==0:
    #     print ("\n  ---interm: \n",data)


print("\n  ---goals: \n", goals)

print("\n  ---final: \n", data)

print("\n  ---final delta: \n", data[:, 0:2] - goals)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# fig, ax = plt.subplots()
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

ax.plot(x,  y,   z, 'ro', linewidth=1)
ax.plot(x2, y2, z, 'ro', color='skyblue')
ax.plot(x3, y3, z, 'ro', color='olive')
ax.plot(x4, y4, z, 'ro', color='yellow')
ax.set(zlabel='time (s)', ylabel='y', xlabel="x",
       title='traj of persons')
ax.grid()

plt.show()
# print ("\n  ---forces: \n",forces)
# print ("forces:", forces)
# print ("init state", state)
# print ("\n    ---delta vel\n", data_copied[:,2:4] - data[:,2:4].view(-1,2))
# print("calcul time: ", time.time()-start)
# print ("\n\n grad \n",grad)
