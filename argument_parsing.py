import argparse

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
    
    parser.add_argument("--input_path", default="./input_data",
        help="Path to folder containing input files.")
    
    parser.add_argument("--l1_reg", type=float, default=0,
        help="Lambda hyperparameter for L1 regularization.")

    parser.add_argument("--l2_reg", type=float, default=0,
        help="Lambda hyperparameter for L2 regularization.")

    parser.add_argument("--learning_rate", type=float, default=0.001,
        help="Learning rate for gradient descent optimization (Adam).")
    
    parser.add_argument("--max_epochs", type=int, default=10000,
        help="Maximum number of training epochs.")
    
    parser.add_argument("--mpnn_aggr", default="mean",
        help="Aggregation function for MPNN messages.")

    parser.add_argument("--mpnn_steps", type=int, default=5,
        help="Number of MPNN message propagation steps.")
    
    parser.add_argument("--mpnn_update", default="mean",
        help="Name of node update function for MPNN layers.")
    
    parser.add_argument("--nb_checkpoints", default=100,
        help="Number of checkpoints at which to save during training (if all epochs run). Only most recent checkpoint is maintained.")
    
    parser.add_argument("--nb_reports", type=int, default=100,
        help="Number of loss reports printed during training (if all epochs run).")

    parser.add_argument("--node_encoding", type=int, default=100,
        help="Length of nodes' hidden encoding vectors.")

    parser.add_argument("--output_path", default="./output",
        help="Path to output directory.")

    parser.add_argument("--properties", default="input_data/properties.csv",
        help="File containing structure names and target values (CSV format).")

    parser.add_argument("--recache", action=argparse.BooleanOptionalAction, default=False,
        help="Ignore and overwrite cache files.")
    
    parser.add_argument("--stalling_threshold", type=float, default=0.0001,
        help="If 3 consecutive validation losses have percent standard deviation less than this value, training stops.")

    parser.add_argument("--stop_threshold", type=float, default=0.1,
        help="Validation loss threshold for early stopping.")

    parser.add_argument("--s2s_steps", type=int, default=5,
        help="Number of processing steps for Set2Set.")

    parser.add_argument("--test_prop", type=float, default=0.2,
        help="Proportion of data to use for testing. Ignored when loading test/train split from disk.")
    
    # process and return arguments
    args = parser.parse_args()
    return args
