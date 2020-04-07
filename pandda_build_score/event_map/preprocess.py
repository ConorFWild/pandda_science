import pandas as pd

def get_train_test_split(event_table,
                         test_split=0.15,
                         num_datasets_bounds=0.05):
    unique_initial_models = event_table["model_dir"].unqiue()

    while True:
        train_tables = []

        train_model_dirs = unique_initial_models.sample(frac=1 - test_split)
        for model_dir in train_model_dirs:
            train_table = event_table[event_table["model_dir"] = model_dir]
            train_tables.append(train_table)

        train_table = pd.concatenate(train_tables)

        if (len(train_table) < test_split + num_datasets_bounds) and (
                len(train_table) > test_split - num_datasets_bounds):
            test_initial_dirs = [initial_dir
                                 for initial_dir
                                 in unique_initial_models
                                 if initial_dir not in train_model_dirs
                                 ]
            test_tables = []
            for test_initial_dir in test_initial_dirs:
                test_table = event_table[event_table["model_dir"] == test_initial_dir]
                test_tables.append(test_table)
            test_table = pd.concatenate(test_tables)
            break

    return train_table, test_table
