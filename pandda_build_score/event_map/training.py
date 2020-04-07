import numpy as np
import pandas as pd

import torch.nn.functional as F
import torch.optim as optim


def train(network,
          dataloader,
          epochs: int = 10,
          ):
    network.train()
    optimizer = optim.SGD(network.parameters(),
                          lr=0.01,
                          )

    for epoch in range(epochs):

        labels = []
        for i_batch, batch in enumerate(dataloader):
            sample_batch = batch["data"]
            label_batch = batch["label"]
            id_batch = batch["id"]

            estimated_label_batch = network(sample_batch)

            loss = F.nll_loss(estimated_label_batch,
                              label_batch,
                              )

            loss.backward()

            print("\t\tBatch {} loss")

            for index, label, estimated_label in zip(id_batch, label_batch, estimated_label_batch):
                record = {"dtag": index["dtag"],
                          "event_idx": index["event_idx"],
                          "true_class": np.argmax(label),
                          "estimated_class": np.argmax(estimated_label),
                          }
                labels.append(record)

        training_table = pd.DataFrame(labels)
        print(training_table.head())

    return network, training_table
