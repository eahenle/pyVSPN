#!/usr/bin/env python3

import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

from helper_functions import cached


def get_node_counts():
    with open("cache/properties.pkl", "rb") as f:
        df = pickle.load(f)

    node_counts = []
    for name in tqdm(df["name"]):
        with open(f"cache/graphs/{name}.pkl", "rb") as f:
            g = pickle.load(f)
        node_counts.append(g.x.shape[0])
    
    return node_counts


def main():
    args = {"cache_path": ".", "recache": False}
    node_counts = cached(lambda : get_node_counts(), "node_counts.pkl", args)

    args = {"cache_path": "cache", "recache": False}
    train, validate, test = cached(lambda : None, "data_split.pkl", args)
    
    plt.hist(node_counts)
    plt.title("Node Count Histogram")
    plt.xlabel("Node Count")
    plt.ylabel("Graphs")
    plt.savefig("node_counts.png")

    print(f"Max Node Count: {max(node_counts)}")
    print(f"Training Set Size: {len(train)}")
    print(f"Validation Set Size: {len(validate)}")
    print(f"Testing Set Size: {len(test)}")

if __name__ == "__main__":
    main()
