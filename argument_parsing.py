import argparse
import random

def parse_args():
    """
    args = parse_args()

    Processes command line arguments and fills in defaults.  Run program with `-h` flag to see arg list.
    """

    # argument parsing object
    parser = argparse.ArgumentParser(
        prog="python main.py", description="MPNN training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # required argument
    parser.add_argument("target", help="Training target column.")
        
    # optional arguments
    parser.add_argument("--batch_size", type=int, default=10,
        help="Number of examples per mini-batch during training. Ignored when loading batches from disk.")
    
    parser.add_argument("--cache_path", default="./cache",
        help="Path to folder containing cache pickle files.")

    parser.add_argument("--device", default="cuda:0",
        help="Device on which to train. Defaults to cuda:0 if available, otherwise cpu.")

    parser.add_argument("--element_embedding", type=int, default=10,
        help="Length of nodes' input layer encoding vectors.")

    parser.add_argument("--hidden_encoding", type=int, default=10,
        help="Length of nodes' hidden encoding vectors.")
    
    parser.add_argument("--input_path", default="./input_data",
        help="Path to folder containing input files.")

    parser.add_argument("--learning_rate", type=float, default=0.001,
        help="Learning rate for gradient descent optimization (Adam).")

    parser.add_argument("--lr_decay_gamma", type=float, default=1.,
        help="Gamma coefficient for learning rate decay.")
    
    parser.add_argument("--max_epochs", type=int, default=10000,
        help="Maximum number of training epochs.")

    parser.add_argument("--model", default="BondingGraphGNN",
        help="Which NN model to use.")
    
    parser.add_argument("--mpnn_aggr", default="add",
        help="Aggregation function for MPNN messages.")

    parser.add_argument("--mpnn_steps", type=int, default=5,
        help="Number of MPNN message propagation steps.")

    parser.add_argument("--output_path", default="./output",
        help="Path to output directory.")

    parser.add_argument("--random_seed", type=int, default=int(42 * random.random()),
        help="Seed for reproducible pseudorandom number generation. Default value is randomly generated.")

    parser.add_argument("--rebound_threshold", type=int, default=10,
        help="Maximum number of consecutive epochs with validation loss increase before early stopping.")

    parser.add_argument("--recache", action=argparse.BooleanOptionalAction, default=False,
        help="Ignore and overwrite cache files.")

    parser.add_argument("--target_data", default="input_data/targets.csv",
        help="File containing structure names and target values (CSV format).")

    parser.add_argument("--test_prop", type=float, default=0.2,
        help="Proportion of data to use for testing. Ignored when loading test/train split from disk.")
    
    parser.add_argument("--val_prop", type=float, default=0.01,
        help="Proportion of data to use for validation. Ignored when loading test/train split from disk.")

    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False,
        help="Toggle printing of detailed information to console.")

    parser.add_argument("--voro_embedding", type=int, default=3,
        help="Length of Voro-node embedding after input layer.")

    parser.add_argument("--voro_h", type=int, default=10,
        help="Hidden representation length for Voro-nodes.")
    
    # process and return arguments
    args = parser.parse_args()
    return args
