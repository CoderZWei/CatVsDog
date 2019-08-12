import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms,datasets
import torchvision.models as models
import numpy as np

data_transform=transforms.Compose([
    transforms.Resize([256,256]),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.4914, 0.4822, 0.4465],
        [0.2023, 0.1994, 0.2010]
    )
    ]
)
test_transform=transforms.Compose([
    transforms.Resize([224,224]),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.4914, 0.4822, 0.4465],
        [0.2023, 0.1994, 0.2010]
    )
    ]
)

train_dataset=datasets.ImageFolder('data/train/',transform=data_transform)
train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)

model=models.resnet50(pretrained=True).cuda()
model_dict=model.state_dict()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.prenet=nn.Sequential()
        self.prenet.add_module('conv1',model.conv1)
        self.prenet.add_module('bn1', model.bn1)
        self.prenet.add_module('relu', model.relu)
        self.prenet.add_module('maxpool', model.maxpool)
        self.prenet.add_module('layer1', model.layer1)
        self.prenet.add_module('layer2', model.layer2)
        self.prenet.add_module('layer3', model.layer3)
        self.prenet.add_module('layer4', model.layer4)

        self.fc1=nn.Linear(100352,512)
        self.drop=nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4=nn.Linear(128,2)

    def forward(self,input):
        x=self.prenet(input)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 100352)
        x = F.relu(self.fc1(x))
        x=self.drop(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

net=Net().cuda()

net1_dict=net.prenet.state_dict()
pretained_dict={k:v for k,v in model_dict.items() if k in net1_dict}
net1_dict.update(pretained_dict)
net.prenet.load_state_dict(net1_dict)
for param in list(net.prenet.parameters()):
    param.requires_grad=False

criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(),lr=0.0001,momentum=0.9)
num_epochs=150

for epoch in range(num_epochs):
    print('epoch:',epoch)
    for i,data in enumerate(train_loader,0):
        img,label=data
        img,label=img.cuda(),label.cuda()
        #img, label = Variable(img), Variable(label)
        optimizer.zero_grad()
        print('label:',label)
        result=net(img)
        loss=criterion(result,label)
        result_label = np.argmax(result.data.cpu().numpy(), axis=1)
        print('result_label:',result_label)
        print(np.sum(result_label == label.cpu().numpy()))
        loss.backward()
        optimizer.step()
    torch.save(net.state_dict(), 'model/params' + str(epoch) + '.pkl')

test_net=Net().cuda()
test_net.load_state_dict(torch.load('model/params87.pkl'))
test_dataset=datasets.ImageFolder('data/test/',transform=test_transform)
test_loader=DataLoader(test_dataset,batch_size=1,shuffle=False)
with torch.no_grad():
    for img,_ in test_loader:
        img=img.cuda()
        img=Variable(img)
        result=test_net(img)
        result_label = np.argmax(result.data.cpu().numpy(), axis=1)
        print(result_label)

