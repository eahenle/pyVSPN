import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        prog="python main.py", description="MPNN training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # required argument
    parser.add_argument("target", help="Training target column")
        
    # optional arguments
    parser.add_argument("--device", default="cuda:0",
        help="Device on which to train. Defaults to cuda:0 if available, otherwise cpu")

    parser.add_argument("--graph_encoding", type=int, default=100,
        help="Length of graph's vector encoding")
    
    parser.add_argument("--l1_reg", type=float, default=0,
        help="Alpha hyperparameter for L1 weight regularization")

    parser.add_argument("--learning_rate", type=float, default=0.001,
        help="Learning rate for gradient descent optimization (Adam)")
    
    parser.add_argument("--max_epochs", type=int, default=10000,
        help="Maximum number of training epochs")

    parser.add_argument("--mpnn_steps", type=int, default=5,
        help="Number of MPNN message propagation steps")
    
    parser.add_argument("--nb_reports", type=int, default=100,
        help="Number of loss reports printed during training (if all epochs run)")

    parser.add_argument("--node_encoding", type=int, default=100,
        help="Length of nodes' hidden encoding vectors")

    parser.add_argument("--properties", default="properties.csv",
        help="File containing structure names and target values (CSV format)")

    parser.add_argument("--stop_threshold", type=float, default=0.1,
        help="Example-averaged loss threshold for early stopping")
    
    # process and return arguments
    args = parser.parse_args()
    return args


def print_args(args):
    print("\nARGS")
    print("########################################")
    arg_dict = vars(args)
    for key in arg_dict:
        print(f"{key}: {arg_dict[key]}")
    print("########################################\n")
