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


from forward import calc_forces, calc_cost_function , calc_new_vel , calc_new_pose, DT


robot_pose = torch.tensor([1.,2.01],requires_grad=True)
goal= torch.tensor([10.,20.],requires_grad=True)
init_pose = torch.tensor([0.,1.],requires_grad=True)
agents_pose = torch.tensor([[1.,3.],[5.,2.],[1.,0.]],requires_grad=True)

na = 4
goals = 5 * torch.rand((na,2),requires_grad=False)


class LinearFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input):
        print ("input ",input)
        forces = calc_forces(input, goals)#.requires_grad_(True)
        print ("forces ", forces)
        data_ = calc_new_vel(input.clone(), forces.clone())
        print ("data_ ", data_)
        data_ = calc_new_pose(data_.clone())
        print ("data_ ", data_)
        # cost = calc_cost_function(agents_pose=data_[:,0:2])
        # print ("cost ", cost)
        ctx.save_for_backward(data_)#, cost)
        return data_

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        print ("here!")
        
        input = ctx.saved_tensors
        print(input[0]) 
        #input,  cost = ctx.saved_tensors
        # grad_input = grad_weight = grad_bias = None
        # print (cost)
        # cost.backward(torch.ones(cost.shape))
        
        # input[0].backward(torch.ones(input[0].shape))
        # grad_input = input[0].grad
        # print ("grad ", grad_input)

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.

        # if ctx.needs_input_grad[0]:
        #     grad_input = grad_output.mm(weight)
        # if ctx.needs_input_grad[1]:
        #     grad_weight = grad_output.t().mm(input)
        # if bias is not None and ctx.needs_input_grad[2]: 
        #     grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input#, grad_weight, grad_bias    

class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        # self.input_features = input_features
        # self.output_features = output_features

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
        #     self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return LinearFunction.apply(input)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


torch.autograd.set_detect_anomaly(True)

rand_data = torch.rand((na,4))
rand_data[:,2:4] *= 0
data = 10 * rand_data
data.requires_grad_(True)

cost = torch.zeros(na,2)

l = Linear()
# new_data = l(data)
print (data)
data = l(data)
print (data)
data.backward(torch.ones(data.shape))
exit()
# print()
# data = l(data)
# print()
# data = l(data)
# print()
# data = l(data)
# new_data.backward(torch.ones(new_data.shape))

# exit()

# forces = calc_rep_forces(data[:,0:2])

# forces = calc_forces(data, goals)
# data_ = calc_new_vel(data.clone(), forces.clone())
# data_ = calc_new_pose(data_.clone())
# cost+=calc_cost_function(agents_pose=data_[:,0:2])
# print ("forces ", forces)
# cost.backward(torch.ones((cost.shape)))

# print (data.grad)#[:,0:2].grad)
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
print ("init state ", data)

for i in range(0,20):

    # forces, _ = l((data,cost))
    
    forces = calc_forces(data, goals)
    # print ("forces ", forces)
    data = calc_new_vel(data, forces)
    # print (data.shape)
    data = calc_new_pose(data)
    data = calc_new_vel(data, forces)
    # print ("new data  ", data)
    cost+=calc_cost_function(agents_pose=data[:,0:2])
    # print (data.shape)
    # print (forces.shape)
    
    
    x.append(data.data[0,0].item())
    y.append(data.data[0,1].item())
    x2.append(data.data[1,0].item())
    y2.append(data.data[1,1].item())
    x3.append(data.data[2,0].item())
    y3.append(data.data[2,1].item())
    x4.append(data.data[3,0].item())
    y4.append(data.data[3,1].item())
    z.append(t)
    t+=DT

# forces.backward(torch.ones((forces.shape)))
print ("final state: ", data)
print ("goals ", goals)
print ("final deltapose: ", data[:,0:2]- goals)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# fig, ax = plt.subplots()
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

ax.plot(x,  y,   z,'ro',linewidth=1)
ax.plot(x2, y2, z,'ro',color='skyblue')
ax.plot(x3, y3, z,'ro',color='olive')
ax.plot(x4, y4, z,'ro',color='yellow')
ax.set(zlabel='time (s)', ylabel='y', xlabel = "x",
       title='traj of persons')
ax.grid()

plt.show()