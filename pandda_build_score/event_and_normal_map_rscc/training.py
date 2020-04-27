from copy import deepcopy

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from pandda_build_score.event_map.functions import get_loss_function


def save_model(network,
               path,
               ):
    torch.save(network.state_dict(),
               str(path),
               )


def evaluate_model(network,
                   dataloader_test,
                   ):
    network.eval()

    records = []
    for i_batch, batch in enumerate(dataloader_test):
        sample_batch = batch["data"]
        label_batch = batch["label"]
        id_batch = batch["id"]
        rscc_class = batch["rscc_class"].type(torch.FloatTensor)
        pandda_name = deepcopy(id_batch["pandda_name"][0])
        dtag = deepcopy(id_batch["dtag"][0])
        event_idx = deepcopy(id_batch["event_idx"][0].detach().numpy())

        estimate_rscc_class = network(sample_batch)

        record = {"pandda_name":pandda_name,
                  "dtag": dtag,
                  "event_idx": event_idx,
                      "rscc_class":  np.argmax(rscc_class),
                  "score": estimate_rscc_class[1],
                  }
        records.append(record)

        del id_batch
        del label_batch
        del batch

        if i_batch == 500:
            break

    test_score_table = pd.DataFrame(records)

    precision_recalls = []
    for i in range(100):
        cutoff = float(i) / 100
        positives_table = test_score_table[test_score_table["score"] > cutoff]
        true_positives_table = positives_table[positives_table["rscc_class"] == 1]
        false_positives_table = positives_table[positives_table["rscc_class"] == 0]

        negatives_table = test_score_table[test_score_table["score"] < cutoff]
        true_negatives_table = negatives_table[negatives_table["rscc_class"] == 0]
        false_negatives_table = negatives_table[negatives_table["rscc_class"] == 1]

        if len(positives_table) == 0:
            precision = 0
        else:
            precision = len(true_positives_table) / len(positives_table)
        if len(true_positives_table) + len(false_negatives_table) == 0:
            recall = 0
        else:
            recall = len(true_negatives_table) / (len(true_positives_table) + len(false_negatives_table))

        precision_recall = {"cutoff": cutoff,
                            "precision": precision,
                            "recall": recall,
                            }
        precision_recalls.append(precision_recall)

    network.train()

    return pd.DataFrame(precision_recalls), test_score_table





def write_results(table,
              path,
              ):
    table.to_csv(str(path))


def train(network,
          dataloader,
          dataloader_test,
          results_dir,
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
            rscc_class = batch["rscc_class"].type(torch.FloatTensor)

            optimizer.zero_grad()

            sample_batch_cuda = sample_batch.cuda()
            label_batch_cuda = label_batch.cuda()
            rscc_batch_cuda = rscc.cuda()
            rscc_class_cuda = rscc_class.cuda()

            estimated_rscc_class_cuda = network(sample_batch_cuda)

            loss = loss_function(estimated_rscc_class_cuda,
                                 rscc_class_cuda,
                                 )

            # loss = loss_function_rscc(estimated_rscc_cuda,
            #                           rscc_batch_cuda,
            #                           )

            # estimated_rscc = estimated_rscc_cuda.cpu()
            estimated_rscc_class = estimated_rscc_class_cuda.cpu()
            # estimated_label = estimated_label_batch_cuda.cpu()
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
                # print("\t{}".format(estimated_label))
                print("\t{}".format(rscc))
                print("\t{}".format(rscc_class))
                print("\t{}".format(estimated_rscc_class))
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
                break

            if i_batch % 1000 == 999:  # print every 30 mini-batches
                print("=======1000 dataset average=======")

                print("\tLoss rscc at epoch {}, iteration {} is {}".format(epoch,
                                                                           i_batch,
                                                                           running_loss_rscc_1000 / 1000))
                running_loss_rscc_3000 = 0

                recent_labels = labels[-999:]

                # correlation = np.mean([np.abs(record["rscc"] - record["estimated_rscc"]) for record in recent_labels if record["rscc"] > 0.5])
                # print("\tCorrelation: {}".format(correlation))
                # positives = [x for x in recent_labels if x["rscc_class"] == 1]
                # negatives = [x for x in recent_labels if x["rscc_class"] == 0]
                # false_positives = [x for x in pos]

            for i, index in enumerate(id_batch["pandda_name"]):
                pandda_name = deepcopy(id_batch["pandda_name"][i])
                dtag = deepcopy(id_batch["dtag"][i])
                event_idx = deepcopy(id_batch["event_idx"][i].detach().numpy())
                class_array = deepcopy(label_batch[i].detach().numpy())
                rscc_array = deepcopy(rscc[i].detach().numpy())
                # estimated_rscc_array = deepcopy(estimated_rscc[i].detach().numpy())
                estimated_rscc_class_array = deepcopy(estimated_rscc_class[i].detach().numpy())
                true_class = np.argmax(class_array)
                event_map_path = deepcopy(batch["event_map_path"][i])
                coords = deepcopy(batch["coords"][i])
                record = {"pandda_name": pandda_name,
                          "dtag": dtag,
                          # "estimated_rscc": estimated_rscc_array,
                          "estimated_rscc_class_array": estimated_rscc_class_array,
                          "rscc": rscc_array,
                          "rscc_class": np.argmax(rscc_class),
                          "event_idx": event_idx,
                          "true_class": true_class,
                          "event_map_path": event_map_path,
                          "coords": coords,
                          }
                labels.append(record)

            del id_batch
            del label_batch
            del batch

        # Save the model
        save_model(network,
                   results_dir / "{epoch}.pt".format(epoch),
                   )

        # Evaluate performance on the test set
        precision_recall_table, test_score_table = evaluate_model(network,
                                                dataloader_test,
                                                )
        print(test_score_table)
        print(precision_recall_table)


        # Update and write table
        write_results(precision_recall_table,
                      results_dir / "precision_recall_{epoch}.csv".format(epoch),
                      )
        write_results(test_score_table,
                      results_dir / "scores_{epoch}.csv".format(epoch),
                      )

        training_table = pd.DataFrame(labels)
        print(training_table.head())

    return network, training_table
