from copy import deepcopy

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from pandda_build_score.event_map.functions import get_loss_function


def train(network,
          dataloader,
          epochs: int = 1000,
          ):
    network.train()
    network.cuda()

    loss_function = get_loss_function()
    loss_function_rscc = nn.MSELoss()

    # optimizer = optim.SGD(network.parameters(),
    #                       lr=0.0001,
    #                       )

    optimizer = optim.Adam(network.parameters(),
                           lr=0.0001,
                           )

    for epoch in range(epochs):

        labels = []

        running_loss_rscc_30 = 0
        running_loss_rscc_100 = 0
        running_loss_rscc_300 = 0
        running_loss_rscc_1000 = 0
        # try:
        for i_batch, batch in enumerate(dataloader):
            sample_batch = batch["data"]
            label_batch = batch["label"]
            id_batch = batch["id"]
            rscc = batch["rscc"].type(torch.FloatTensor)

            optimizer.zero_grad()

            sample_batch_cuda = sample_batch.cuda()
            rscc_batch_cuda = rscc.cuda()

            estimated_label_batch_cuda, estimated_rscc_cuda = network(sample_batch_cuda)

            loss = loss_function_rscc(estimated_rscc_cuda,
                                           rscc_batch_cuda,
                                           )


            estimated_rscc = estimated_rscc_cuda.cpu()
            loss.backward()
            optimizer.step()


            running_loss_rscc_30 += loss.item()
            running_loss_rscc_100 += loss.item()
            running_loss_rscc_300 += loss.item()
            running_loss_rscc_1000 += loss.item()

            if i_batch % 30 == 29:  # print every 30 mini-batches
                print("=30 dataset average=")
                print("\tLoss rscc at epoch {}, iteration {} is {}".format(epoch,
                                                                      i_batch,
                                                                      running_loss_rscc_30 / 30))
                print("\t{}".format(id_batch))
                print("\t{}".format(label_batch))
                print("\t{}".format(rscc))
                print("\t{}".format(estimated_rscc))
                running_loss_rscc_30 = 0

            if i_batch % 100 == 99:  # print every 30 mini-batches
                print("===100 dataset average===")

                print("\tLoss rscc at epoch {}, iteration {} is {}".format(epoch,
                                                                      i_batch,
                                                                      running_loss_rscc_100 / 100))
                running_loss_rscc_100 = 0


            if i_batch % 300 == 299:  # print every 30 mini-batches
                print("=====300 dataset average=====")

                print("\tLoss rscc at epoch {}, iteration {} is {}".format(epoch,
                                                                      i_batch,
                                                                      running_loss_rscc_300 / 300))
                running_loss_rscc_300 = 0


            if i_batch % 1000 == 999:  # print every 30 mini-batches
                print("=======1000 dataset average=======")

                print("\tLoss rscc at epoch {}, iteration {} is {}".format(epoch,
                                                                      i_batch,
                                                                      running_loss_rscc_1000 / 1000))
                running_loss_rscc_3000 = 0


                recent_labels = labels[-999:]

                correlation = np.sum([record["rscc"] / record["estimated_rscc"] for record in recent_labels])
                print("\tCorrelation: {}".format(correlation))


            for i, index in enumerate(id_batch["pandda_name"]):
                pandda_name = deepcopy(id_batch["pandda_name"][i])
                dtag = deepcopy(id_batch["dtag"][i])
                event_idx = deepcopy(id_batch["event_idx"][i].detach().numpy())
                class_array = deepcopy(label_batch[i].detach().numpy())
                rscc_array = deepcopy(rscc[i].detach().numpy())
                estimated_rscc_array = deepcopy(estimated_rscc[i].detatch().numpy())
                true_class = np.argmax(class_array)
                event_map_path = deepcopy(batch["event_map_path"][i])
                coords = deepcopy(batch["coords"][i])
                record = {"pandda_name": pandda_name,
                          "dtag": dtag,
                          "estimated_rscc": estimated_rscc_array,
                         "rscc": rscc_array,
                          "event_idx": event_idx,
                          "true_class": true_class,
                          "event_map_path": event_map_path,
                          "coords": coords,
                          }
                labels.append(record)

            del id_batch
            del label_batch
            del batch



        training_table = pd.DataFrame(labels)
        print(training_table.head())

    return network, training_table
