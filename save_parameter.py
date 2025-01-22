if __name__ == '__main__':
    import argparse
    import numpy as np
    from utils import *
    import time
    from model import *
    import os
    from torchsummary import summary
    import torch

    # Parse input arguments
    parser = argparse.ArgumentParser(description='Save S-ResNet Parameters', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', type=str, help='path	 to model')
    parser.add_argument('--arch',                  default='sresnet', type=str, help='architecture used by the model')
    parser.add_argument('--n',                     default=6, type=int, help='Depth scaling of the S-ResNet')
    parser.add_argument('--nFilters',              default=32, type=int, help='Width scaling of the S-ResNet')
    parser.add_argument('--boosting',              default=False, action='store_true', help='Use boosting layer')
    parser.add_argument('--dataset', default='cifar100', type=str, help='Dataset [cifar10, cifar100, cifar10dvs]')
    parser.add_argument('--batch_size',            default=500,       type=int,   help='Batch size')
    parser.add_argument('--num_steps',             default=50,    type=int, help='Number of time-step')
    parser.add_argument('--leak_mem', default=0.874, type=float, help='Leak_mem')
    parser.add_argument('--device', default=None, type=int, help='gpu number to use')
    parser.add_argument('--seed', default=1234, type=int, help='random seed')
    parser.add_argument('--poisson_gen',default=False, action='store_true', help='use poisson spike generation')

    global args
    args = parser.parse_args()

    if args.device is not None:
        torch.cuda.set_device(args.device)

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Define model

    leak_mem = args.leak_mem
    batch_size      = args.batch_size
    num_steps       = args.num_steps

    # Instantiate the SNN model and optimizer
    num_cls = 10
    img_size = 32

    if args.arch == 'sresnet':
        model = SResnet(n=args.n, nFilters=args.nFilters, num_steps=num_steps, leak_mem=leak_mem, img_size=img_size, num_cls=num_cls,
                        boosting=args.boosting, poisson_gen=args.poisson_gen)
    elif args.arch == 'sresnet_nm':
        model = SResnetNM(n=args.n, nFilters=args.nFilters, num_steps=num_steps, leak_mem=leak_mem, img_size=img_size, num_cls=num_cls)

    else:
        print("Architecture name not found")
        exit()

    # Load weights
    model_dict = torch.load(args.model_path, map_location='cpu')
    state_dict = model_dict['state_dict']
    reload_epoch = model_dict['global_step']
    best_acc = model_dict['accuracy']
    own_state = model.state_dict()

    # # Directory where the parameter files are saved
    # output_dir = "model_cifar10_50t_parameters"

    # # Initialize an empty state_dict to load the parameters
    # state_dict = {}

    # # Iterate over all files in the directory
    # for file_name in os.listdir(output_dir):
    #     if file_name.endswith(".txt") and file_name != "summary.txt":
    #         # Split the file name into parts
    #         parts = file_name.split("_")

    #         # The parameter name is everything before the last underscore
    #         param_name = "_".join(parts[:-1])

    #         # The shape is the last part (after the last underscore) without the .txt extension
    #         shape_str = parts[-1].replace(".txt", "")

    #         # Convert the shape string into a tuple of integers
    #         param_shape = tuple(map(int, shape_str.split(".")))

    #         # Construct the full file path
    #         file_path = os.path.join(output_dir, file_name)

    #         # Load the parameter values from the file
    #         with open(file_path, "r") as f:
    #             param_values = np.loadtxt(f)  # Read values as a 1D array

    #         # Reshape the parameter values to the original shape
    #         param_values = param_values.reshape(param_shape)

    #         print(f"Processing file: {file_name}")
    #         print(f"Extracted parameter name: {param_name}")
    #         print(f"Extracted shape: {param_shape}")
    #         print(f"Loaded values shape: {param_values.shape}")

    #         # Convert the NumPy array to a PyTorch tensor and add it to the state_dict
    #         state_dict[param_name] = torch.tensor(param_values)

    # # Load the model's state_dict
    # own_state = model.state_dict()

    # # Copy the loaded parameters into the model's state_dict
    # for name, param in state_dict.items():
    #     if name in own_state:
    #         own_state[name].copy_(param)
    #     else:
    #         print(f"Skipping: {name}")

    # # Print a summary of the loaded weights
    # print("Reloaded weights from saved files.")

    # Set NumPy print options to avoid truncation
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)

    # Create a directory to store the parameter files
    output_dir = "model_mnist_parameters_test"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize counters
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0

    # Iterate over all parameters in the model
    for name, param in model.named_parameters():
        param_count = param.numel()  # Get the total number of elements in the parameter tensor
        total_params += param_count

        if param.requires_grad:
            trainable_params += param_count
        else:
            non_trainable_params += param_count

        # Create a file name based on the parameter name and shape
        file_name = f"{name}_{'.'.join(map(str, param.data.size()))}.txt"
        file_path = os.path.join(output_dir, file_name)

        # Write the parameter values to the file (one value per line)
        with open(file_path, "w") as f:
            param_values = param.data.cpu().numpy().flatten()  # Flatten the array to 1D
            for value in param_values:
                f.write(f"{value}\n")  # Write each value on a new line

    # Write summary to a separate file
    summary_file_path = os.path.join(output_dir, "summary.txt")
    with open(summary_file_path, "w") as f:
        f.write("Summary:\n")
        f.write(f"Total parameters: {total_params}\n")
        f.write(f"Trainable parameters: {trainable_params}\n")
        f.write(f"Non-trainable parameters: {non_trainable_params}\n")

    print(f"All parameters have been saved to the directory: {output_dir}")


