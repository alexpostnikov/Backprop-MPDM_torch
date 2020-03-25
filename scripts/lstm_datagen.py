from social_lstm.predict import *
import time
import torch
pass




# TODO 
'''
class dataloader

get as params numb of peds, obs_len

method update_scene:
    save new incoming(observed) poses of peds
    from torch to list(ndarray(ndarray(float)))
    if num of observed poses > obs_len:
        forget first, add new (circullar buffer)
    
method get_batch:
    x = list(ndarray(ndarray(float)))
    return : x, y, d , numPedsList, PedsList ,target_ids

'''

class Lstm_Datagen(Datagen):
    def __init__(self,num_peds, pred_length=8, obs_length=20, ):
        Datagen.__init__(self, pred_length, obs_length)
        
        self.num_peds = num_peds
        pose = np.zeros(3)
        peds_poses = np.zeros([self.num_peds, 3])
        self.observed_poses_num = 0
        self.obs_poses = np.zeros([self.seq_length,self.num_peds,3])
    
    @property
    def is_redy_to_predict(self):
        return self.observed_poses_num >= self.obs_length
    
    def update_scene(self, state):
        state = self.extract_poses_from_state(state)
        if self.observed_poses_num <= self.obs_length-1:
            self.obs_poses[self.observed_poses_num] = state
            self.observed_poses_num += 1
        
        else:
            self.obs_poses[0:self.obs_length-1] = self.obs_poses[1:self.obs_length]
            self.obs_poses[self.obs_length-1] = state

            

    def extract_poses_from_state(self, state):
        poses = np.zeros([self.num_peds,3])
        for i,person_state in enumerate(state):
            poses[i] = np.array([ np.append(i,person_state[0:2].cpu().detach().numpy())])
        return poses


    def next_batch(self):
        # numPedsList -> list with numb of peds at each timestemp
        # Pedlist -> list of peds indexes at each timestemp
        # targget_ids -> id of pedestrian target (to do moved to 0,0 pose?) 
        # x_seq -> list(20 x timestemps(each timestemp list with number of peds (with poses and ident for each ped))
        # numPedsList = list(i for i in)
        assert (self.observed_poses_num >= self.obs_length)

        PedsList = [[list(range(self.num_peds))]*(self.pred_length + self.obs_length)]
        numPedsList =  [list((len(i)for i in PedsList[0]))]
        target_ids = [0]
        d = [0]
        x = [list(i for i in range(self.pred_length + self.obs_length))]
        for timestemp in range(len(x[0])):
            x[0][timestemp] = np.zeros([self.num_peds,3])
            if timestemp < self.obs_length:
                x[0][timestemp] = self.obs_poses[timestemp]
        y = None
        return x, y, d, numPedsList, PedsList ,target_ids
    
    def fake_data_gen(self):
        # produce fake data (testing purposes mainly)

        assert (self.pred_length + self.obs_length) > 0
        assert (self.numpeds>0)
        
        # inner_levels = {"timestamp":1 , "ped_number":2,"pose":3}

        x = [list(i for i in range(self.pred_length + self.obs_length))]
        for timestamp in range(self.pred_length + self.obs_length):
            x[0][timestamp] = np.zeros([self.numpeds,3])
            for ped_number in range(self.numpeds):
                x_pose = y_pose = ped_number*2
                if timestamp > 0:
                    x_pose = x[0][timestamp-1][ped_number][1] #- 0.2 - 0.5*np.random.rand()
                    y_pose = x[0][timestamp-1][ped_number][2] - 0.2 - 0.5*np.random.rand()
                x[0][timestamp][ped_number] = np.array([ped_number,x_pose,y_pose])

        return x


if __name__ == "__main__":
    
    ## TEST ## 

    datagen = Lstm_Datagen(10)
    for i in range (0,10):
        a = 10* torch.rand(10,4)
        datagen.update_scene(a)
    # datagen.next_batch()

    sample_args = Sample_args(pl=8,ol=20)
    net,saved_args = get_net(sample_args)
    st = time.time()
    process(datagen,net,sample_args,saved_args, 0) 

    # print ("inf time: ", time.time()- st)

