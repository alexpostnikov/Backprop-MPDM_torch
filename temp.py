import torch

DT = 0.2
a = 5
b = 2
e = 0.001
robot_speed = 0.1
robot_pose = torch.tensor([1.,2.01],requires_grad=True)
goal= torch.tensor([10.,20.],requires_grad=True)
init_pose = torch.tensor([0.,1.],requires_grad=True)
agents_pose = torch.tensor([[1.,3.],[5.,2.],[1.,0.]],requires_grad=True)


def tensor_inf_to_zero(bad_tensor):
    bad_tensor[bad_tensor == float('inf')] = 0
    bad_tensor.requires_grad_(True)
    bad_tensor.retain_grad()
    return bad_tensor

def calc_forces(state):
    rep_force = 1 * calc_rep_forces(state)
    attr_force =1 *  calc_attractive_forces(state)
    return rep_force + attr_force

def calc_attractive_forces(input):
    global goals
    delta_pose = goals -input
    # print ("delta_pose ", delta_pose)
    dist = torch.sqrt( delta_pose[:,0:1]**2 + delta_pose[:,1:2]**2)
    force = delta_pose/torch.cat((dist,dist),dim=1)
    return force 


def calc_cost_function(a=a, b=b,e=e,robot_speed=robot_speed, robot_pose=robot_pose,goal=goal,init_pose=init_pose,agents_pose=agents_pose ):
    costs = torch.zeros(agents_pose.shape,requires_grad=False)
    costs.retain_grad()
    PG = (robot_pose - init_pose)*(-init_pose+goal)/torch.norm(-init_pose+goal)
    PG.retain_grad()
    # Blame
    B = torch.zeros(len(agents_pose),requires_grad=False)
    B.retain_grad()
    if robot_speed>e:
        for n in range(len(agents_pose)):
            # TODO: go into matrix math
            B[n] = torch.exp(-torch.norm(agents_pose[n]-robot_pose)/b)
    # Cost
    for n in range(len(B)):
        costs[n] = -a*PG+B[n]
    return costs


def limit_speed(input_state, limit):

    ampl = torch.sqrt(input_state[:,0:1]**2 + input_state[:,1:2]**2)
    
    mask=torch.cat((ampl>limit,ampl>limit),dim=1)
    ampl_2D = torch.cat((ampl,ampl),dim=1)
    # print (mask)
    # print (input_state[mask])
    # print (ampl_2D[ampl_2D>3])
    input_state[mask] = input_state[mask].clone() * limit /ampl_2D[mask]
    return input_state

def calc_new_vel(input_state, forces):
    # input_vel = input_state[:,2:4].view(-1,2)
    input_state[:,2:4] = (forces * DT / 6.0 )+ input_state[:,2:4]
    # limit max speed : 
    # input_state[:,2:3]**2 + input_state[:,3:4]**2
    input_state[:,2:4] = limit_speed(input_state[:,2:4], 1)
    return input_state

def calc_new_pose(input):
    input[:,0:2] = input[:,0:2] + input[:,2:4] * DT
    return input

def calc_rep_forces(state):
    # state = state_[:,0:2]

    aux1 = torch.tensor(([1.,0.],[0.,1.])) # used to transform state from [N_rows*2] to [N_rows*(2*N_rows)]
    for i in range(0,state.shape[0]-1):
        aux1 = torch.cat((aux1, torch.tensor(([1.,0.],[0.,1.]))),dim=1)
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
    state_concated = state.matmul(aux1)
    
    state_concated.retain_grad()
    # print ("state ", state)
    state_concated_t = state.reshape(1,-1)#.requires_grad_(True)
    for i in range(0,state.shape[0]-1):
        state_concated_t = torch.cat([state_concated_t,state.reshape(1,-1)])#.requires_grad_(True)
        '''    state_concated_t tensor(
        [[ 1.,  0.,  2.,  1., -1., -1.,  0., -1.],
         [ 1.,  0.,  2.,  1., -1., -1.,  0., -1.],
         [ 1.,  0.,  2.,  1., -1., -1.,  0., -1.],
         [ 1.,  0.,  2.,  1., -1., -1.,  0., -1.]]
        '''
    state_concated_t#.requires_grad_(True)
    delta_pose = (state_concated_t - state_concated)#.requires_grad_(True)
    # delta_pose.retain_grad()
    
    auxullary = torch.zeros(state_concated.shape[0], state_concated.shape[1])
    # auxullary.retain_grad()
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
    dist = (dist_squared.matmul(aux)).requires_grad_(True) ## aka distance
    dist = torch.sqrt(dist) + 10000*torch.eye(dist.shape[0])  #TODO: deal with 1/0,
    A = 2 * (10**3) # const param from  formula(21) from `When Helbing Meets Laumond: The Headed Social Force Model`
    force_amplitude = A * torch.exp((0.3 -dist)/0.08).requires_grad_(True) ## according to Headed Social Force Model
    
    force = force_amplitude.matmul(delta_pose/(dist+0.00001).matmul(auxullary)) # formula(21) from `When Helbing Meets Laumond: The Headed Social Force Model`
    force.requires_grad_(True)
    force.retain_grad()    
    aux2 = aux1.clone().t()
    force = force.matmul(aux2).requires_grad_(True)
    return force


def test(state):
    res = state.matmul(torch.ones((2,2)))
    return res

na = 4
rand_data = torch.rand((na,4))
data = 10 * rand_data
data.requires_grad_(True)
# forces = calc_rep_forces(data[:,0:2])
print ("data", data)
goals =20 * torch.rand((na,2),requires_grad=False)
# print ("forces ", forces)
# forces.backward(torch.ones((forces.shape)))

# print (data.grad)#[:,0:2].grad)

cost = torch.zeros(na,2)

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
for i in range(0,5000):

    # forces, _ = l((data,cost))
    forces = calc_forces(data[:,0:2])
    data = calc_new_vel(data, forces)
    # print (data.shape)
    data = calc_new_pose(data)
    cost+=calc_cost_function(agents_pose=data[:,0:2])
    # print (data.shape)
    # print (forces.shape)
    
    data = calc_new_vel(data, forces)
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