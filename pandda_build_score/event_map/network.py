import torch.nn as nn
import torch.nn.functional as F


class NetworkDep(nn.Module):
    def __init__(self,
                 shape,
                 ):
        super(Network, self).__init__()
        self.shape = shape

        self.activation_function = nn.Softmax()

        self.bn = nn.BatchNorm3d(3)

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
        self.pool1 = nn.MaxPool3d(kernel_size=2,
                                  stride=2,
                                  )
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
        self.pool2 = nn.MaxPool3d(kernel_size=2,
                                  stride=2,
                                  )
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
        self.pool3 = nn.MaxPool3d(kernel_size=2,
                                  stride=2,
                                  )
        self.dropout3 = nn.Dropout3d(p=0.5)

        self.fc1 = nn.Linear(int(self.shape[0] * self.shape[1] * self.shape[2] * (1 / 8) * (1 / 8) * (1 / 8) * 35),
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
                   x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4],
                   )
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = self.activation_function(x)

        return x


class NetworkDep2(nn.Module):
    def __init__(self,
                 shape,
                 ):
        super(Network, self).__init__()
        self.shape = shape

        self.activation_function = nn.Softmax()

        self.block1 = nn.Sequential(nn.Conv3d(1,
                                              10,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              ),
                                    nn.BatchNorm3d(10),
                                    nn.ReLU(),
                                    nn.Conv3d(10,
                                              15,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              ),
                                    nn.BatchNorm3d(15),
                                    nn.ReLU(),
                                    nn.MaxPool3d(kernel_size=2,
                                                 stride=2,
                                                 ),
                                    nn.Dropout3d(p=0.5),
                                    )

        self.block2 = nn.Sequential(nn.Conv3d(15,
                                              20,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              ),
                                    nn.BatchNorm3d(20),
                                    nn.ReLU(),
                                    nn.Conv3d(20,
                                              25,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              ),
                                    nn.BatchNorm3d(25),
                                    nn.ReLU(),
                                    nn.MaxPool3d(kernel_size=2,
                                                 stride=2,
                                                 ),
                                    nn.Dropout3d(p=0.5),
                                    )

        self.block3 = nn.Sequential(nn.Conv3d(25,
                                              30,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              ),
                                    nn.BatchNorm3d(30),
                                    nn.ReLU(),
                                    nn.Conv3d(30,
                                              35,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              ),
                                    nn.BatchNorm3d(35),
                                    nn.ReLU(),
                                    nn.MaxPool3d(kernel_size=2,
                                                 stride=2,
                                                 ),
                                    nn.Dropout3d(p=0.5),
                                    )

        self.fc1 = nn.Linear(int(self.shape[0] * self.shape[1] * self.shape[2] * (1 / 8) * (1 / 8) * (1 / 8) * 35),
                             30,
                             )
        self.fc2 = nn.Linear(30,
                             2,
                             )

    def forward(self, x):
        # Perform the usual forward pass
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = x.view(-1,
                   x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4],
                   )
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = self.activation_function(x)

        return x


class ResidualBlock(nn.Module):

    def __init__(self, filters_out):
        # Instatiate Network layers

        super().__init__()

        self.conv_1 = nn.Conv3d(filters_out, filters_out, kernel_size=3, stride=1, padding=1)

        self.bn = nn.BatchNorm3d(filters_out)

        self.act = nn.ReLU()

    def forward(self, x):
        residual = self.conv_1(x)
        residual = self.bn(residual)

        x = x + residual

        return self.act(x)


class ResidualLayerWithDrop(nn.Module):

    def __init__(self, filters_in, filters_out, training=True):
        # Instatiate Network layers

        super().__init__()

        self.conv_1 = nn.Sequential(nn.Conv3d(filters_in, filters_out, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm3d(filters_out),
                                    nn.ReLU())

        self.res_1 = ResidualBlock(filters_out)
        # self.res_2 = ResidualBlock(filters_out)

        self.drop = nn.Dropout3d(p=0.5)

        self.mp = nn.MaxPool3d(kernel_size=2, stride=2)
        self.do_drop = training

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.res_1(x)
        # x = self.res_2(x)
        if self.training:
            x = self.drop(x)
        x = self.mp(x)

        return self.act(x)


class Network(nn.Module):
    """
    CNN for recognising interesting ligands
    1x introductory convolution, 3x residual layers of 2, 2x
    No dropout
    No global pooling
    """

    def __init__(self, filters=10, grid_size=32, training=True):
        # Instatiate Network layers

        super().__init__()

        self.filters = filters
        self.grid_size = grid_size

        self.bn = nn.BatchNorm3d(1)

        self.conv_1 = nn.Sequential(nn.Conv3d(1, filters, kernel_size=9, stride=1, padding=4),
                                    nn.BatchNorm3d(filters),
                                    nn.ReLU(),
                                    nn.MaxPool3d(kernel_size=2, stride=2))

        self.layer_1 = ResidualLayerWithDrop(filters, filters * 2, training)
        self.layer_2 = ResidualLayerWithDrop(filters * 2, filters * 4, training)
        self.layer_3 = ResidualLayerWithDrop(filters * 4, filters * 8, training)

        size_after_convs = int((grid_size / (2 ** 4)) ** 3)
        n_in = filters * 8 * size_after_convs

        self.fc1 = nn.Sequential(nn.Linear(n_in, int(n_in / 4)),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(int(n_in / 4), 2))

        self.act = nn.Softmax()

    def forward(self, x):
        x = self.conv_1(x)

        x = self.layer_1(x)

        x = self.layer_2(x)

        x = self.layer_3(x)

        x = x.view(-1, (x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4]))

        x = self.fc1(x)

        x = self.fc2(x)

        return self.act(x)


def get_network(shape):
    network = Network(10, grid_size=32)

    return network
