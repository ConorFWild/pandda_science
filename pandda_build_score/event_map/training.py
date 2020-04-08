import numpy as np
import pandas as pd

from torch import nn
import torch.nn.functional as F
import torch.optim as optim


def train(network,
          dataloader,
          epochs: int = 10,
          ):
    network.train()
    loss_function = nn.BCELoss()

    optimizer = optim.SGD(network.parameters(),
                          lr=0.00001,
                          )
    # optimizer = optim.Adam(network.parameters(),
    #                        lr=0.00001)

    for epoch in range(epochs):

        labels = []
        running_loss = 0
        for i_batch, batch in enumerate(dataloader):
            sample_batch = batch["data"]
            label_batch = batch["label"]
            id_batch = batch["id"]

            sample_batch = sample_batch.unsqueeze(1)

            estimated_label_batch = network(sample_batch)

            # loss = F.nll_loss(estimated_label_batch,
            #                   label_batch,
            #                   )
            loss = loss_function(estimated_label_batch,
                                 label_batch,
                                 )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i_batch % 30 == 29:  # print every 100 mini-batches
                print("\tLoss at epoch {}, iteration {} is {}".format(epoch,
                                                                      i_batch,
                                                                      running_loss / 30) + "\n")
                print(estimated_label_batch)
                print(label_batch)
                running_loss = 0
            # print("\t\tBatch {} loss")

            for i, index in enumerate(id_batch["pandda_name"]):
                record = {"pandda_name": id_batch["pandda_name"][i],
                          "dtag": id_batch["dtag"][i],
                          "event_idx": id_batch["event_idx"][i],
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

        training_table = pd.DataFrame(labels)
        print(training_table.head())

    return network, training_table
