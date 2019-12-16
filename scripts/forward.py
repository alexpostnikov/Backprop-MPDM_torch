import torch
from force_attract_with_dest import k, force_goal, pose_propagation, DT, pedestrians_speed, is_goal_achieved, generate_new_goal

a = 5
b = 2
e = 0.001
rep_coef = 1.
robot_speed = 1

robot_pose = torch.tensor([1.,2.01],requires_grad=True)
goal= torch.tensor([10.,20.],requires_grad=True)
init_pose = torch.tensor([0.,1.],requires_grad=True)
agents_pose = torch.tensor([[1.,3.],[5.,2.],[1.,0.]],requires_grad=True)

def calc_forces(state, goals):
    rep_force = calc_rep_forces(state[:,0:2])
    # rep_force = torch.clamp(rep_force,min=-5., max = 5.)
    attr_force =  force_goal(state, goals)
    return rep_force + attr_force



def calc_cost_function(a=a, b=b,e=e,robot_speed=robot_speed, robot_pose=robot_pose,goal=goal,init_pose=init_pose,agents_pose=agents_pose ):
    costs = torch.zeros(agents_pose.shape,requires_grad=False)
    # costs.retain_grad()
    PG = (robot_pose - init_pose)*(-init_pose+goal)/torch.norm(-init_pose+goal)
    # PG.retain_grad()
    # Blame
    B = torch.zeros(len(agents_pose),requires_grad=False)
    # B.retain_grad()
    if robot_speed>e:
        for n in range(len(agents_pose)):
            # TODO: go into matrix math
            B[n] = torch.exp(-torch.norm(agents_pose[n]-robot_pose)/b)
    # Cost
    for n in range(len(B)):
        costs[n] = -a*PG+B[n]
    return costs


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
    state_concated = state.clone().matmul(aux1)
    state_concated.retain_grad()
    state_concated_t = state.reshape(1,-1)
    for i in range(0,state.shape[0]-1):
        state_concated_t = torch.cat([state_concated_t,state.reshape(1,-1)])
        '''    state_concated_t tensor(
        [[ 1.,  0.,  2.,  1., -1., -1.,  0., -1.],
         [ 1.,  0.,  2.,  1., -1., -1.,  0., -1.],
         [ 1.,  0.,  2.,  1., -1., -1.,  0., -1.],
         [ 1.,  0.,  2.,  1., -1., -1.,  0., -1.]]
        '''
    delta_pose = (-state_concated_t + state_concated)
    
    auxullary = torch.zeros(state_concated.shape[0], state_concated.shape[1])
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
    dist_squared = ((delta_pose)**2)
    dist_squared.retain_grad()
    # used to calc delta_x**2 +delta_y**2 of each agent
    aux = auxullary.t()
    aux.retain_grad()
    # sqrt(delta_x**2 +delta_y**2) -> distance
    dist_squared += 0.0000001 # TODO: otherwise  when doing backprop: sqrt(0)' -> nan
    dist = (dist_squared.matmul(aux))
     ## aka distance
    dist = torch.sqrt(dist) + 10000000*torch.eye(dist.shape[0])  #TODO: deal with 1/0,
    # print ("dist: ", dist)
    A = 2 * (10**3) # const param from  formula(21) from `When Helbing Meets Laumond: The Headed Social Force Model`
    force_amplitude = A * torch.exp((0.3 - dist) / 0.08) ## according to Headed Social Force Model
    # print ("force_amplitude: ", force_amplitude)

    # print ("delta_pose / (dist).matmul(auxullary) \n",delta_pose / (dist).matmul(auxullary))
    force = force_amplitude.matmul(auxullary)*(delta_pose / (dist).matmul(auxullary)) # formula(21) from `When Helbing Meets Laumond: The Headed Social Force Model`
    # print ("force BBbefore: ", force)
    force = (force * ((auxullary -1) *-1))
    # print ("force before: ", force)
    aux2 = aux1.clone().t()
    force = force.matmul(aux2)
    # print ("force after: ", force)
    # print ()
    return force


if __name__ == "__main__":

    torch.autograd.set_detect_anomaly(True)
    na = 2
    rand_data = torch.rand((na,4))
    # rand_data[:,2:4] *= 0
    data = rand_data
    
    data.requires_grad_(True)
    goals = torch.rand((na,2),requires_grad=False)
    cost = torch.zeros(na,2)


    # forces = calc_forces(data, goals)
    # data_ = pose_propagation(forces,data)
    # cost+=calc_cost_function(agents_pose=data_[:,0:2])
    # print (cost)
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
    for i in range(0, 1):
        forces = calc_forces(data, goals)
        
        data = pose_propagation(forces,data)
        cost+=calc_cost_function(agents_pose=data[:,0:2])
        
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
        res = is_goal_achieved(data, goals)
        if any(res) == True:
            goals = generate_new_goal(goals,res, data)

    # forces.backward(torch.ones((forces.shape)))
    # print ("final state: ", data)
    # print ("goals ", goals)
    # print ("final deltapose: ", data[:,0:2]- goals)
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