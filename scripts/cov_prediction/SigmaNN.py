import torch
import numpy as np

class SigmaNN:
    def __init__(self, model_path="./model.pth"):
        self.model = None
        try:
            self.model = torch.load(model_path)
            print("SigmaNN model success loaded by path: " + model_path)
        except:
            print("can`t load SigmaNN model by path: "+model_path)
        pass
    
    def calc_covariance(self, cov_prev, state_prev, state_cur):
        out = []
        for agents_num in range(len(state_cur)):
            input_ = torch.from_numpy(np.stack((cov_prev[agents_num], state_prev[agents_num],state_cur[agents_num]))).reshape(6,1).float()
            cov_ = self.model(input_.T).detach().numpy()[0]
            out.append(cov_.tolist())
        return out

if __name__ == "__main__":
    print("test SigmaNN class")
    model = SigmaNN()