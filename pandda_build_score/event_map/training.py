import numpy as np
import pandas as pd

from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from pandda_build_score.event_map.functions import get_loss_function

def train(network,
          dataloader,
          epochs: int = 10,
          ):
    network.train()

    loss_function = get_loss_function()

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
        try:
            for i_batch, batch in enumerate(dataloader):
                sample_batch = batch["data"]
                label_batch = batch["label"]
                id_batch = batch["id"]
                sample_batch = sample_batch.unsqueeze(1)

                optimizer.zero_grad()

                estimated_label_batch = network(sample_batch)
                loss = loss_function(estimated_label_batch,
                                     label_batch,
                                     )
                loss.backward()
                optimizer.step()

                running_loss_30 += loss.item()
                running_loss_100 += loss.item()
                running_loss_300 += loss.item()


                if i_batch % 30 == 29:  # print every 30 mini-batches
                    print("\tLoss at epoch {}, iteration {} is {}".format(epoch,
                                                                          i_batch,
                                                                          running_loss_30 / 30) + "\n")
                    print(estimated_label_batch)
                    print(label_batch)
                    running_loss_30 = 0

                if i_batch % 100 == 99:  # print every 30 mini-batches
                    print("\tLoss at epoch {}, iteration {} is {}".format(epoch,
                                                                          i_batch,
                                                                          running_loss_100 / 30) + "\n")
                    print(estimated_label_batch)
                    print(label_batch)
                    running_loss_100 = 0

                if i_batch % 300 == 299:  # print every 30 mini-batches
                    print("\tLoss at epoch {}, iteration {} is {}".format(epoch,
                                                                          i_batch,
                                                                          running_loss_300 / 30) + "\n")
                    print(estimated_label_batch)
                    print(label_batch)
                    running_loss_300 = 0

                for i, index in enumerate(id_batch["pandda_name"]):
                    record = {"pandda_name": id_batch["pandda_name"][i],
                              "dtag": id_batch["dtag"][i],
                              "event_idx": id_batch["event_idx"][i].detach().numpy(),
                              "true_class": np.argmax(label_batch[i].detach().numpy()),
                              "estimated_class": np.argmax(estimated_label_batch[i].detach().numpy()),
                              }
                    # print(record)
                # for index, label, estimated_label in zip(id_batch, label_batch, estimated_label_batch):
                #     print(index)
                #     print(label)
                #     print(estimated_label)
                #     record = {"dtag": index["dtag"],
                #               "event_idx": index["event_idx"],
                #               "true_class": np.argmax(label),
                #               "estimated_class": np.argmax(estimated_label),
                #               }
                    labels.append(record)
        except Exception as e:
            print("Failed for some reason")
            print(e)
        training_table = pd.DataFrame(labels)
        print(training_table.head())

    return network, training_table
