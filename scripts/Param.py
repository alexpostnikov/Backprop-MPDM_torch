import torch

class Param:
    def __init__(self):
        # ros
        self.loop_rate = 30.
        
        self.num_ped = 3
        self.optim_epochs = 5
        self.number_of_layers = 10
        self.do_visualization = 1
        self.do_logging = 0

        self.area_size = 3
        self.pedestrians_speed = 1.0
        self.robot_init_pose = torch.tensor(([1.5,2.0]))
        self.look_ahead_seconds = 4

        # mpdm params
        self.k = 2.3
        self.DT = 0.4
        self.alpha = 10.66
        
        self.ped_mass = 60
        self.betta = 0.71

        # social force params
        self.socForceRobotPerson = {"k":2.3, "lambda":0.59, "A":2.66, "B":0.79,"d":0.5}
        self.socForcePersonPerson = {"k":4.9, "lambda":1., "A":10., "B":0.64,"d":0.16}
        # self.socForceRobotPerson = {"k":1.3, "lambda":0.59, "A":2.66, "B":0.79,"d":0.5}
        # self.socForcePersonPerson = {"k":2.9, "lambda":1., "A":10., "B":0.64,"d":0.16}

        self.a = 0.001
        self.b = 3
        self.e = 0.001
        self.robot_speed = 1

        
        # self.update_scene(new_pose_mean, new_goal_mean)
        

        self.generateMatrices()
        
        self.init_calcs()
        self.robot_goal = self.goal[0,2:4]

    def update_scene(self, new_pose_mean, new_goal_mean):
        self.input_state_mean = new_pose_mean
        self.input_distrib = torch.distributions.normal.Normal(self.input_state_mean, self.input_state_std)
        self.input_state = self.input_state_mean

        self.goal_mean = new_goal_mean
        
        self.goal_distrib = torch.distributions.normal.Normal(self.goal_mean, self.goal_std)
        self.goal = self.goal_mean
        self.goal = self.goal.view(-1, 2)

    def generateMatrices(self):
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
        self.goal_mean = goals
        self.goal = goals
        return goals

    def is_goal_achieved(self, state, goals):
        is_achieved = state[:,0:2] - goals
        is_achieved = torch.sqrt(is_achieved[:,0]**2 + is_achieved[:,1]**2)
        return is_achieved<0.3

    def init_calcs(self):
        self.loop_sleep = 1/self.loop_rate
        self.goal_mean = self.area_size*torch.rand((self.num_ped,2))
        self.goal_std = 2.0 * torch.rand((self.num_ped,2))
        self.goal_distrib = torch.distributions.normal.Normal(self.goal_mean, self.goal_std)

        self.goal = self.goal_mean
        self.goal = self.goal.view(-1, 2)
        
        self.input_state_mean = self.area_size*torch.rand((self.num_ped,4))
        self.input_state_mean[:,2:4] = self.input_state_mean[:,2:4]/ self.area_size
        
        self.input_state_std = 1.0 * torch.rand((self.num_ped,4))
        self.input_distrib = torch.distributions.normal.Normal(self.input_state_mean, self.input_state_std)

        #self.input_state = self.input_distrib.sample()
        self.input_state = self.input_state_mean
        self.input_state[0,0:2] = self.robot_init_pose#.clone().detach()
        self.input_state = self.input_state.view(-1, 4).requires_grad_(True)

        # self.goal[0,:] = self.robot_goal.clone().detach()
        self.robot_init_pose = self.robot_init_pose.clone().detach().requires_grad_(True)
        # self.robot_goal= torch.tensor(self.robot_goal,requires_grad=True)


if __name__ == "__main__":
    p = Param()
    print ("p.input_state_mean", p.input_state_mean)
    print ("p.input_state_std ", p.input_state_std)
    print (p.input_state)
    m = torch.distributions.normal.Normal(torch.tensor([0.0, 5.2]), torch.tensor([1.0, 0.02]))
    t = m.sample()  # normally distributed with loc=0 and scale=1
    print (t)
    





# 	//force_params_to_vector(double k, double lamb,double A,double B,double d);

# 	switch( person_force_type_ )
# 	{
# 	case Collision_Prediction:
# 		// Zanlungo collision prediction parameters
# 		set_social_force_parameters_person_person( force_params_to_vector(1.52, 0.29,1.13,0.71,0.0) );
# 		//Person-Robot Spherical parameters obtained using our optimization method
# 		set_social_force_parameters_person_robot( force_params_to_vector(1.52, 0.29,1.13,0.71,0.0) );
# 		//set_social_force_parameters_person_robot( force_params_to_vector(2.3, 0.59,2.66,0.79,0.4) );
# 		//Obstacle spherical parameters obtained using our optimization method
# 		set_social_force_parameters_obstacle( force_params_to_vector(2.3, 1.0,10.0,0.1,0.2) );
# 		break;
# 	case Elliptical:
# 		//Default Zanlungo Elliptical parameters
# 		set_social_force_parameters_person_person( force_params_to_vector(1.19, 0.08,1.33,0.34,1.78) );
# 		//Person-Robot Spherical parameters obtained using our optimization method
# 		set_social_force_parameters_person_robot( force_params_to_vector(2.3, 0.59,2.66,0.79,0.4) );
# 		//Obstacle spherical parameters obtained using our optimization method
# 		set_social_force_parameters_obstacle( force_params_to_vector(2.3, 1.0,10.0,0.1,0.2) );
# 		break;
# 	case Spherical:
# 		//Default Zanlungo Spherical parameters (2.3, 0.08,1.33,0.64,0.16)
# 		set_social_force_parameters_person_person( force_params_to_vector(4.9, 1.0,10.0,0.64,0.16) );//B=0.34, changed to 0.64
# 		//Person-Robot Spherical parameters obtained using our optimization method
# 		set_social_force_parameters_person_robot( force_params_to_vector(2.3, 0.59,2.66,0.79,0.4) );
# 		//Obstacle spherical parameters obtained using our optimization method
# 		set_social_force_parameters_obstacle( force_params_to_vector(2.3, 1.0,10.0,0.1,0.2) );
# 		break;
# 	}


# }