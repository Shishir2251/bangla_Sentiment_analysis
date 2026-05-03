import pandas as pd
from config import DATASET_NAME, DATA_PATHS

def load_data():
    base_path = DATA_PATHS[DATASET_NAME]

    train = pd.read_csv(base_path + "/train.tsv", sep="\t")
    dev = pd.read_csv(base_path + "/dev.tsv", sep="\t")
    test = pd.read_csv(base_path + "/test.tsv", sep="\t")


    return train, dev, test