import torch

class Param:
    def __init__(self):
        # ros
        self.loop_rate = 30.
        
        self.area_size = 15
        self.num_ped = 10

        self.pedestrians_speed = 1.0
        self.robot_init_pose = torch.tensor(([1.5,2.0]))
        self.robot_goal = torch.tensor(([10.,20.]))
        self.robot_speed = 2.0
        
        self.look_ahead_seconds = 4

        
        # mpdm params
        self.k = 2.3
        self.DT = 0.2
        self.alpha = 10.66
        
        self.ped_mass = 60
        self.betta = 0.71
        # robot params

        # social force params
        self.socForceRobotPerson = {"k":1.3, "lambda":0.59, "A":2.66, "B":0.79,"d":0.5}
        self.socForcePersonPerson = {"k":2.9, "lambda":1., "A":10., "B":0.64,"d":0.16}
        
        self.a = 10
        self.b = 2
        self.e = 0.001
        self.robot_speed = 1

        self.do_visualization = 1

        self.generateMatricesAB()
        self.init_calcs()



    def generateMatricesAB(self):
        self.alpha = self.socForcePersonPerson["A"] * (1 - torch.eye(self.num_ped,self.num_ped))
        self.alpha[0,:] = self.socForceRobotPerson["A"]
        self.alpha[:,0] = self.socForceRobotPerson["A"]

        self.betta = self.socForcePersonPerson["B"] * torch.ones(self.num_ped,self.num_ped)
        self.betta[0,:] = self.socForceRobotPerson["B"]
        self.betta[:,0] = self.socForceRobotPerson["B"]

        self.k = self.socForcePersonPerson["k"] * torch.ones(self.num_ped)
        self.k[0] = self.socForceRobotPerson["k"]
        self.k  = self.k.view(-1,1)

        self.ped_radius = self.socForcePersonPerson["d"] * torch.ones(self.num_ped,self.num_ped)
        self.ped_radius[0,:] = self.socForceRobotPerson["d"]
        self.ped_radius[:,0] = self.socForceRobotPerson["d"]

        self.lamb = self.socForcePersonPerson["lambda"] * torch.ones(self.num_ped,self.num_ped)
        self.lamb[0,:] = self.socForceRobotPerson["lambda"]
        self.lamb[:,0] = self.socForceRobotPerson["lambda"]


    def generate_new_goal(self, goals, input_state):
        is_achived = self.is_goal_achieved(input_state, goals)
        if any(is_achived) == True:
            for i in range (is_achived.shape[0]):
                if is_achived[i].item() == True:
                    goals[i,0] = self.area_size*torch.rand(1)
                    goals[i,1] = self.area_size*torch.rand(1)
        return goals

    def is_goal_achieved(self, state, goals):
        is_achieved = state[:,0:2] - goals
        is_achieved = torch.sqrt(is_achieved[:,0]**2 + is_achieved[:,1]**2)
        return is_achieved<0.3

    def init_calcs(self):
        self.loop_sleep = 1/self.loop_rate
        self.goal = self.area_size*torch.rand((self.num_ped,2))
        self.goal = self.goal.view(-1, 2)
        self.input_state = self.area_size*torch.rand((self.num_ped,4))
        self.input_state[:,2:4] = self.input_state[:,2:4]/ self.area_size
        self.input_state = self.input_state.view(-1, 4)
        self.input_state[0,0:2] = self.robot_init_pose.clone().detach()
        self.goal[0,:] = self.robot_goal.clone().detach()
        self.robot_init_pose = self.robot_init_pose.clone().detach().requires_grad_(True)
        # self.robot_goal= torch.tensor(self.robot_goal,requires_grad=True)


if __name__ == "__main__":
    p = Param()