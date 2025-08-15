import torch
import torch.nn as nn 


class IMG_CLASSIFIER_MODEL(nn.Module):
  
  def __init__(self,input_size,hidden_size,output_size):
    super().__init__()
    self.block = nn.Sequential(
      nn.Conv2d(in_channels=input_size,out_channels=hidden_size,kernel_size=3,stride=1,padding=0),
      nn.ReLU(),
      nn.Conv2d(in_channels=hidden_size,out_channels=hidden_size,kernel_size=3,stride=1,padding=0),
      nn.MaxPool2d((2,2))
    ) 

    with torch.no_grad():
      dummy = torch.zeros(1,input_size,64,64)
      dummy_out = (self.block(dummy))
      flattened_size = dummy_out.view(1, -1).size(1)
    
    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Linear(in_features=flattened_size,out_features=output_size)
    )
  
  def forward(self,x):
    x = self.block(x)
    x = self.classifier(x)
    return x