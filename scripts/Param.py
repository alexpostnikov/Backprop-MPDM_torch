import torch

class Param:
    def __init__(self):
        # ros
        self.loop_rate = 100.
        
        # mpdm params
        self.k = 2.2
        self.DT = 0.002
        self.pedestrians_speed = 0.5
        self.num_ped = 5
        self.area_size = 4

        # 
        self.a = 5
        self.b = 2
        self.e = 0.001
        self.robot_speed = 1


        self.init_calcs()


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
        self.input_state = self.input_state.view(-1, 4)