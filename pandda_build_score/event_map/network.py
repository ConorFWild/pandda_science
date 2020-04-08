import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self,
                 shape,
                 ):
        super(Network, self).__init__()
        self.shape = shape
        self.conv1 = nn.Conv3d(1,
                               10,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               )
        self.conv2 = nn.Conv3d(10,
                               20,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               )
        self.conv3 = nn.Conv3d(20,
                               1,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               )
        self.fc1 = nn.Linear(self.shape[0]*self.shape[1]*self.shape[2],
                             20,
                             )
        self.fc2 = nn.Linear(20,
                             2,
                             )

    def forward(self, x):
        activation_function = nn.Softmax()
        # Perform the usual forward pass
        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))

        x = F.relu(self.conv3(x))

        x = x.view(-1,
                   self.shape[0]*self.shape[1]*self.shape[2],
                   )
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = activation_function(x)

        return x


def get_network(shape):
    network = Network(shape=shape)

    return network
