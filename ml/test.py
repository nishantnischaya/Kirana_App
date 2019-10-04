import os
import cv2
#from training.py import ConvNet
import numpy as  np
import torch
from torchvision import transforms as trans
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import matplotlib.pyplot as p
from torch.utils.data import TensorDataset as dset
from torch.utils.data import DataLoader as dl
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2,2 ),
            nn.ReLU())
       
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(30 * 40 * 32, 3125)
        self.fc2 = nn.Linear(3125, 125)
        self.fc3 =nn.Linear(125,5)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = f.relu(self.fc1(out))
        out = f.relu(self.fc2(out))
        out=f.relu(self.fc3(out))
        return out

#
print("hello")
#x=np.load("testdata.npy",allow_pickle=True)

#m=torch.stack([(torch.from_numpy(i[0]))for i in x])
#n=torch.stack([(torch.from_numpy(i[1]))for i in x])    
#myset=dset(m,n)
#print(x.type)
#test_loader=dl(myset,batch_size=64,shuffle=True)
model = ConvNet()
model.load_state_dict(torch.load("./model.pth"))
print("load")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
def webcam():
    h=0
def argmax(image):
    image= cv2.resize(image,(160,120))
    images=torch.from_numpy(image)
     
    outputs = model(images.view(-1,3,120,160).type(torch.FloatTensor))
    probability=torch.nn.Softmax()(outputs)
    print(probability)
    #type error may exist with cuda change to torch.cuuda .Floattensor
    _, predicted = torch.max(outputs.data, 1)
    return predicted
argmax(cv2.imread(r"C:\Users\prath\Desktop\New folder (2)\Test\colgate (115).jpg"))
