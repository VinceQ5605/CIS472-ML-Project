import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, nout):
        super(AlexNet, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0)
        nn.init.kaiming_uniform_(self.conv_1.weight)
        self.conv_2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        nn.init.kaiming_uniform_(self.conv_2.weight)
        self.conv_3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_uniform_(self.conv_3.weight)
        self.conv_4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_uniform_(self.conv_4.weight)
        self.conv_5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_uniform_(self.conv_5.weight)
        self.linear_1 = nn.Linear(in_features=9216, out_features=4096)
        self.linear_2 = nn.Linear(in_features=4096, out_features=4096)
        self.linear_3 = nn.Linear(in_features=4096, out_features=nout)
        self.relu = nn.ReLU(True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.drop = nn.Dropout()
        
    def forward(self, x):
        x = self.relu(self.conv_1(x))
        x = self.pool(x)
        x = self.relu(self.conv_2(x))
        x = self.pool(x)
        x = self.relu(self.conv_3(x))
        x = self.relu(self.conv_4(x))
        x = self.relu(self.conv_5(x))
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.relu(self.linear_1(x))
        x = self.drop(x)
        x = self.relu(self.linear_2(x))
        x = self.linear_3(x)
        return x