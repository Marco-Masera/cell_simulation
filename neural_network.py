import torch
from torch import nn
import torch.nn.functional as F
from params import *

class CellNeuralNetwork(nn.Module):
    def __init__(self):
        super(CellNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(cell_memory*2, 6, 5)  # Input channels = 3, output channels = 6, kernel size = 5
        nn.init.xavier_uniform_(self.conv1.weight) 
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling over a (2, 2) window
        self.conv2 = nn.Conv2d(6, 16, 5)  # Input channels = 6, output channels = 16, kernel size = 5
        nn.init.xavier_uniform_(self.conv2.weight) 
        self.conv3 = nn.Conv2d(16, 10, 5, stride=2)  # Input channels = 16, output channels = 10, kernel size = 5
        nn.init.xavier_uniform_(self.conv3.weight) 
        output_size = 23
        self.size_of_fclayers = output_size**2 * 10
        self.fc1 = nn.Linear(self.size_of_fclayers+2, 10)  # Fully connected layer, output size = 15
        nn.init.xavier_uniform_(self.fc1.weight) 
        self.fc2 = nn.Linear(10, 10)  # Fully connected layer, output size = 10
        nn.init.xavier_uniform_(self.fc1.weight) 
        self.cuda()
        self.random_change()

    def forward(self, x, lifespan, size):
        lifespan = torch.tensor([lifespan / cell_lifespan, float(size / chess_size)]).cuda()
        lifespan = lifespan.reshape(1,2)# = lifespan.repeat(1, 4840).view(-1, 4840)
        x = self.pool(F.relu(self.conv1(x)))  # Pass data through conv1, apply ReLU activation function, then max pooling
        x = self.pool(F.relu(self.conv2(x)))  # Pass data through conv2, apply ReLU activation function, then max pooling
        x = self.pool(F.relu(self.conv3(x)))
        # Reshape data to input to the fully connected layer
        x = x.view(-1, self.size_of_fclayers)
        #Add lifespan to the input as another element in the dimension 1
        x = torch.cat((x, lifespan), dim=1)
        x = F.relu(self.fc1(x))  # Pass data through fc1, apply ReLU activation function
        x = self.fc2(x)  # Pass data through fc3
        return x
    
    def random_change(self):
        for param in self.parameters():
            param.data += torch.randn(param.size()).cuda() * network_changing_rate

    def get_copy(self):
        n = CellNeuralNetwork().cuda()
        #Copy the weights into new network
        n.load_state_dict(self.state_dict())
        #Apply random change to the weights
        n.random_change()
        return n