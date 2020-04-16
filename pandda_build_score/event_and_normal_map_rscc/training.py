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
        running_loss_30 = 0
        running_loss_100 = 0
        running_loss_300 = 0
        running_loss_1000 = 0
        # try:
        for i_batch, batch in enumerate(dataloader):
            sample_batch = batch["data"]
            label_batch = batch["label"]
            id_batch = batch["id"]
            rscc = batch["rscc"].type(torch.FloatTensor)

            # print(sample_batch.shape)

            optimizer.zero_grad()

            sample_batch_cuda = sample_batch.cuda()
            label_batch_cuda = label_batch.cuda()

            estimated_label_batch_cuda, estimated_rscc_cuda = network(sample_batch_cuda)
            # loss = loss_function(estimated_label_batch_cuda,
            #                      label_batch_cuda,
            #                      )
            loss = loss_function_rscc(estimated_rscc_cuda,
                                           rscc,
                                           )
            print(estimated_rscc_cuda.type())
            print(rscc.type())

            # total_loss = loss + loss_rscc

            estimated_label_batch = estimated_label_batch_cuda.cpu()
            estimated_rscc = estimated_rscc_cuda.cpu()
            loss.backward()
            optimizer.step()
            #
            # loss_rscc.backward()
            # optimizer.step()

            # total_loss.backward()
            # optimizer.step()

            running_loss_30 += loss.item()
            running_loss_100 += loss.item()
            running_loss_300 += loss.item()
            running_loss_1000 += loss.item()

            if i_batch % 30 == 29:  # print every 30 mini-batches
                print("=30 dataset average=")
                print("\tLoss at epoch {}, iteration {} is {}".format(epoch,
                                                                      i_batch,
                                                                      running_loss_30 / 30))
                print("\t{}".format(id_batch))
                print("\t{}".format(estimated_label_batch))
                print("\t{}".format(label_batch))
                print("\t{}".format(rscc))
                print("\t{}".format(estimated_rscc))
                running_loss_30 = 0

            if i_batch % 100 == 99:  # print every 30 mini-batches
                print("===100 dataset average===")
                print("\tLoss at epoch {}, iteration {} is {}".format(epoch,
                                                                      i_batch,
                                                                      running_loss_100 / 100))
                running_loss_100 = 0

            if i_batch % 300 == 299:  # print every 30 mini-batches
                print("=====300 dataset average=====")
                print("\tLoss at epoch {}, iteration {} is {}".format(epoch,
                                                                      i_batch,
                                                                      running_loss_300 / 300))
                running_loss_300 = 0

            if i_batch % 1000 == 999:  # print every 30 mini-batches
                print("=======1000 dataset average=======")
                print("\tLoss at epoch {}, iteration {} is {}".format(epoch,
                                                                      i_batch,
                                                                      running_loss_1000 / 1000))
                running_loss_1000 = 0

                recent_labels = labels[-999:]

                for cutoff in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
                    true_positives = 0
                    false_positives = 0
                    true_negatives = 0
                    false_negatives = 0

                    for label in recent_labels:
                        if label["estimated_class"] >= cutoff:
                            if label["true_class"] == 1:
                                true_positives += 1
                            else:
                                false_positives += 1
                        else:
                            if label["true_class"] == 1:
                                false_negatives += 1
                            else:
                                true_negatives += 1
                    print("\tAt cutoff: {}".format(cutoff))
                    print("\t\tThe number of true positives is: {}".format(true_positives))
                    print("\t\tThe number of false positives is: {}".format(false_positives))
                    print("\t\tThe number of false negatives is: {}".format(false_negatives))
                    print("\t\tThe number of true Negatives is: {}".format(true_negatives))
                    if true_positives+false_positives > 0:
                        print("\t\tPrecision is: {}".format(true_positives/(true_positives+false_positives)))
                    else:
                        print("\t\tPrecision is: {}".format(0))
                    if true_positives+false_negatives > 0:
                        print("\t\tRecall is: {}".format(true_positives/(true_positives+false_negatives)))
                    else:
                        print("\t\tRecall is: {}".format(0))

                table = pd.DataFrame(recent_labels)
                table = table[table["true_class"] == 0]

                print([(row["pandda_name"],
                        row["dtag"],
                        row["event_idx"],
                        row["estimated_class"],
                        row["coords"],
                        row["event_map_path"],) for index, row in table.sort_values(by="estimated_class").tail(10).iterrows()])

            for i, index in enumerate(id_batch["pandda_name"]):
                pandda_name = deepcopy(id_batch["pandda_name"][i])
                dtag = deepcopy(id_batch["dtag"][i])
                event_idx = deepcopy(id_batch["event_idx"][i].detach().numpy())
                class_array = deepcopy(label_batch[i].detach().numpy())
                true_class = np.argmax(class_array)
                estimated_class_array = deepcopy(estimated_label_batch[i].detach().numpy())
                estimated_class = estimated_class_array[1]
                event_map_path = deepcopy(batch["event_map_path"][i])
                coords = deepcopy(batch["coords"][i])
                record = {"pandda_name": pandda_name,
                          "dtag": dtag,
                          "event_idx": event_idx,
                          "true_class": true_class,
                          "estimated_class": estimated_class,
                          "event_map_path": event_map_path,
                          "coords": coords,
                          }
                labels.append(record)

            del id_batch
            del label_batch
            del estimated_label_batch
            del batch


        # except Exception as e:
        #     print("Failed for some reason")
        #     print(e)
        training_table = pd.DataFrame(labels)
        print(training_table.head())

    return network, training_table
