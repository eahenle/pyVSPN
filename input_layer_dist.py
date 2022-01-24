import random
import matplotlib.pyplot as plt

from model_definition import EmbeddingBlock
from data_handling import load_data
from argument_parsing import parse_args
from helper_functions import choose_device, check_paths, print_args, write_args, prepare_cache


def main():
    # get command line args (or defaults)
    args = parse_args()
    # display the args
    print_args(args)

    # check for required folders
    check_paths(args)

    # write the args as text in the output folder
    write_args(args)

    # select training device (CPU/GPU)
    device = choose_device(args)

    # set random seed
    random.seed(args.random_seed)

    # prep cache
    prepare_cache(args)

    # load data as DataLoader objects
    training_data, validation_data, test_data, atom_feature_length, voro_feature_length = load_data(device, args)
    # extract voro feature tensor from first test batch
    data = [batch for batch in test_data][0].x_v

    # embedding sizes to test
    embedding_lengths = [i+1 for i in range(16)]
    # loop over embedding sizes
    for voro_embedding_length in embedding_lengths:
        # instantiate the model w/ specified embedding size
        model = EmbeddingBlock(voro_feature_length, voro_embedding_length)
        # calculate embeddings and record the mean value for each column
        embeddings = model(data).detach().numpy()
        # plot heatmap (for embedding length) of element value vs. column index
        fig, ax = plt.subplots()
        plt.hist(embeddings, range=[0, 1])
        plt.title(f"Embedding Length {voro_embedding_length}")
        plt.xlabel("Vector Component Value")
        plt.ylabel("Frequency")
        plt.savefig(f"heatmap_{voro_embedding_length}.png")


__name__ == "__main__" and main()
