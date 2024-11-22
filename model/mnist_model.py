import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1, bias = False) # param =  #input -? OUtput? RF
        self.batchN1 = nn.BatchNorm2d(32)


        self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias = False) # param =
        self.batchN2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)


        self.conv3 = nn.Conv2d(32, 16, 3, padding=1, bias = False) # param =
        self.batchN3 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(16, 16, 3, bias = False) #

        self.conv5 = nn.Conv2d(16, 16, 3, bias = False)

        self.gap1 = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(16, 10)

        self.drop = nn.Dropout2d(0.05)


    def forward(self, x):
        #L 1
        #print("x shape1", x.shape)
        x = self.batchN1(F.relu(self.conv1(x)))
        x = self.drop(x)

        # L 2
        #print("x shape2", x.shape)
        x = self.pool2(self.batchN2(F.relu(self.conv2(x))))
        x = self.drop(x)

        # L 3
        #print("x shape3", x.shape)
        x = F.relu(self.conv3(x))
        #print("x shape4", x.shape)
        self.batchN3(x)
        #print("x shape5", x.shape)
        x = self.pool3(x)

        # L 4
        #print("x shape6", x.shape)
        x = F.relu(self.conv4(x))

        # L 5
        #print("x shape6", x.shape)
        x = F.relu(self.conv5(x))

        # L 6
        #print("x shape1", x.shape)
        x = self.gap1(x)

        #print("x shape2", x.shape)
        x = x.view(-1, 16)

        #print("x shape9", x.shape)
        x = self.fc1(x)

        #print("\nx shape4", x.shape)
        return F.log_softmax(x, dim = 1)