import torch
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self, hidden_features, nout, dropout=False):
        super().__init__()
        self.dropout = dropout
        self.conv_1a = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), padding=1)
        nn.init.kaiming_uniform_(self.conv_1a.weight)
        self.conv_1b = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=1)
        nn.init.kaiming_uniform_(self.conv_1b.weight)
        self.conv_2a = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=1)
        nn.init.kaiming_uniform_(self.conv_2a.weight)
        self.conv_2b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding=1)
        nn.init.kaiming_uniform_(self.conv_2b.weight)
        self.conv_3a = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), padding=1)
        nn.init.kaiming_uniform_(self.conv_3a.weight)
        self.conv_3b = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=1)
        nn.init.kaiming_uniform_(self.conv_3b.weight)
        self.conv_4a = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), padding=1)
        nn.init.kaiming_uniform_(self.conv_4a.weight)
        self.conv_4b = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding=1)
        nn.init.kaiming_uniform_(self.conv_4b.weight)
        self.relu = nn.ReLU(True)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear_1 = nn.Linear(512*14*14, hidden_features)
        self.linear_2 = nn.Linear(hidden_features, hidden_features)
        self.linear_3 = nn.Linear(hidden_features, nout)
        self.drop = nn.Dropout()
        self.batchnorm_64 = nn.BatchNorm2d(64)
        self.batchnorm_128 = nn.BatchNorm2d(128)
        self.batchnorm_256 = nn.BatchNorm2d(256)
        self.batchnorm_512 = nn.BatchNorm2d(512)
        
    def forward(self, x):
        x = self.relu(self.conv_1a(x))
        x = self.relu(self.conv_1b(x))
        x = self.pool(x) # 64x16x16
        x = self.batchnorm_64(x)
        x = self.relu(self.conv_2a(x))
        x = self.relu(self.conv_2b(x))
        x = self.pool(x) # 128x8x8
        x = self.batchnorm_128(x)
        x = self.relu(self.conv_3a(x))
        x = self.relu(self.conv_3b(x))
        x = self.relu(self.conv_3b(x))
        x = self.pool(x) # 256x4x4
        x = self.batchnorm_256(x)
        x = self.relu(self.conv_4a(x))
        x = self.relu(self.conv_4b(x))
        x = self.relu(self.conv_4b(x))
        x = self.pool(x) # 512x2x2
        x = self.batchnorm_512(x)
        x = x.view(x.shape[0], -1)
        x = self.relu(self.linear_1(x))
        if self.dropout:
            x = self.drop(x)
        x = self.relu(self.linear_2(x))
        x = self.linear_3(x)
        return x