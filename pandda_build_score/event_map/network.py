import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self,
                 shape,
                 ):
        super(Network, self).__init__()
        self.shape = shape

        self.activation_function = nn.Softmax()

        self.conv1 = nn.Conv3d(1,
                               10,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               )
        self.conv2 = nn.Conv3d(10,
                               15,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               )
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout3d(p=0.5)

        self.conv3 = nn.Conv3d(15,
                               20,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               )
        self.conv4 = nn.Conv3d(20,
                               25,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               )
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout3d(p=0.5)

        self.conv5 = nn.Conv3d(25,
                               30,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               )
        self.conv6 = nn.Conv3d(30,
                               35,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               )
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout3d(p=0.5)

        self.fc1 = nn.Linear(self.shape[0]*self.shape[1]*self.shape[2]*(1/2)*(1/2)*(1/2)*35,
                             30,
                             )
        self.fc2 = nn.Linear(30,
                             2,
                             )

    def forward(self, x):
        # Perform the usual forward pass
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)
        x = self.dropout3(x)

        x = x.view(-1,
                   x.shape[1]*x.shape[2]*x.shape[3]*x.shape[4],
                   )
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = self.activation_function(x)

        return x


def get_network(shape):
    network = Network(shape=shape)

    return network
