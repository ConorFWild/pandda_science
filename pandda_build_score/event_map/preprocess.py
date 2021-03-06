import numpy as np
import pandas as pd


def get_train_test_split(event_table,
                         test_split=0.15,
                         num_datasets_bounds=0.05):
    unique_initial_models = event_table["model_dir"].unique()

    while True:
        train_tables = []

        # train_model_dirs = unique_initial_models.sample(frac=1 - test_split)
        train_model_dirs = np.random.choice(unique_initial_models,
                                            size=int((1 - test_split) * len(unique_initial_models)),
                                            replace=False,
                                            )
        for model_dir in train_model_dirs:
            train_table = event_table[event_table["model_dir"] == model_dir]
            train_tables.append(train_table)

        train_table = pd.concat(train_tables)

        print("Number of unique initial dirs: {}".format(len(unique_initial_models)))
        print("Number of sampled systems: {}".format(len(train_model_dirs)))
        print("Train table length: {}".format(len(train_table)))
        print("target length: {}".format((1 - test_split) * len(event_table)))
        print("Target upper bound: {}".format(((1 - test_split) + num_datasets_bounds) * len(event_table)))
        print("Target lower bound: {}".format(((1 - test_split) - num_datasets_bounds) * len(event_table)))

        if (len(train_table) < ((1 - test_split) + num_datasets_bounds) * len(event_table)) and (
                len(train_table) > ((1 - test_split) - num_datasets_bounds) * len(event_table)):

            test_initial_dirs = [initial_dir
                                 for initial_dir
                                 in unique_initial_models
                                 if initial_dir not in train_model_dirs
                                 ]
            test_tables = []
            for test_initial_dir in test_initial_dirs:
                test_table = event_table[event_table["model_dir"] == test_initial_dir]
                test_tables.append(test_table)
                test_table = pd.concat(test_tables)
            break

    return train_table, test_table
