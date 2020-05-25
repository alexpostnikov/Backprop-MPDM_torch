import torch

class SigmaNN:
    def __init__(self, model_path="./model.pth"):
        self.model = None
        try:
            self.model = torch.load(model_path)
            print("SigmaNN model success loaded by path: " + model_path)
        except:
            print("can`t load SigmaNN model by path: "+model_path)
        pass

if __name__ == "__main__":
    print("test SigmaNN class")
    model = SigmaNN()