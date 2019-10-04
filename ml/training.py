import os
import cv2
import numpy as  np
from tqdm import tqdm
import torch
from torchvision import transforms as trans
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import matplotlib.pyplot as p
from torch.utils.data import TensorDataset as dset
from torch.utils.data import DataLoader as dl
classes=5
batchsize=16
mlr=0.001

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


#tdset=myset()
x=np.load("traindata.npy",allow_pickle=True)

m=torch.stack([(torch.from_numpy(i[0]))for i in x])
n=torch.stack([(torch.from_numpy(i[1]))for i in x])    
myset=dset(m,n)
#print(x.type)
tload=dl(myset,batch_size=batchsize,shuffle=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ConvNet()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=mlr)
total_step = len(tload)
loss_list = []
acc_list = []
for epoch in range(3):# while loss <x can also be used
    for i, (images, labels) in tqdm(enumerate(tload)):
        # Run the forward pass
        images.to(device)
        labels.to(device)
        #print(images.shape)
        outputs = model(images.view(-1,3,120,160).type(torch.FloatTensor))
    #print(outputs.shape)
    #print(labels.shape)
    #print(labels)
        loss = criterion(outputs,  np.argmax(labels,axis=1))
        loss_list.append(loss.item())
        print(loss)

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
    #print(predicted)
        correct = (predicted ==  np.argmax(labels,axis=1)).sum().item()
        acc_list.append(correct / total)
        print(correct/total)
       
#print(model.state_dict())
torch.save(model.state_dict(),"./model.pth")
#print("hello")
x=np.load("testdata.npy",allow_pickle=True)

m=torch.stack([(torch.from_numpy(i[0]))for i in x])
n=torch.stack([(torch.from_numpy(i[1]))for i in x])    
myset=dset(m,n)
#print(x.type)
test_loader=dl(myset,batch_size=16,shuffle=False)
#model = ConvNet()
#model.load_state_dict(torch.load("./model.pt"))
#print("load")
model.eval()
#print("stage1")
with torch.no_grad():
    #print("stage 2")
    correct = 0
    total = 0
    for images, labels in test_loader:
        #print("stage3")
        print(images.shape)
        outputs = model(images.view(-1,3,120,160).type(torch.FloatTensor))
        #type error may exist with cuda change to torch.cuuda .Floattensor
        probability=torch.nn.Softmax()(outputs)
        print(probability)
        print(labels)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == np.argmax(labels,axis=1)).sum().item()
        #print("running")

    print('Test Accuracy of the model on the  test images: {} %'.format((correct / total) * 100))




#data= next(iter(tload))
#print(data[0].shape)

#l=data[0].view(100,100)
#p.imshow(l,cmap='gray') 
#p.show()

#p.imshow(tdata[1][0])
#p.show()
