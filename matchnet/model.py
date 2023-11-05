import torch.nn as nn

class MatchNet(nn.Module):
    def __init__(self):
        super(MatchNet, self).__init__()
        self.fc1 = nn.Linear(6,32)
        self.fc2 = nn.Linear(32,128)
        self.fc3 = nn.Linear(128,32)
        self.fc4 = nn.Linear(32,1)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x




