import argparse
import os
from pprint import pformat
import torch
import numpy as np
import random

from steps import train, infer
from misc.utils import Logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ENVIRONMENT
    parser.add_argument('--gpus', default=list(range(torch.cuda.device_count())), type=int, nargs='+', help='IDs of the GPUs to use for training.')
    # Argument to fix seeds and foster experiment reproducibility. If None (default), no seed is set.
    parser.add_argument('--experiment_seed', default = None, type = int, 
                        help = "Seed for torch, numpy and random libraries to reduce random effect and allow training reproducibility. If None (default), no seed is set.")

    # DATASET
    # Image folders
    parser.add_argument('--train_image_folder', required=True, type=str,
                        help='Path to the directory containing the images used for training')
    parser.add_argument('--val_image_folder', required=True, type=str,
                        help='Path to the directory containing the images used for validation')
    parser.add_argument('--infer_image_folder', required=True, type=str,
                        help='Path to the directory containing the images used for inference.')
    # Set description file
    parser.add_argument('--train_desc_file', default=None, type=str,
                        help='Path to the file with information about train set images. Needed during training step.')
    parser.add_argument('--val_desc_file', default=None, type=str,
                        help='Path to the file with information about validation set images. Needed during training step.')
    parser.add_argument('--infer_list_file', default=None, type=str,
                        help='Path to the file with the list of images for inference.')

    # TRAINING PARAMS
    parser.add_argument('--network', default=None, type=str, help="The name of the network to train or to use for inference.")            
    parser.add_argument('--batch_size', default=16, type=int, help="The batch size to adopt during training.")
    parser.add_argument('--num_epochs', default=50, type=int, help="The number of epochs to adopt during training.")
    parser.add_argument('--learning_rate', default=0.05, type=float, help="The learning rate to adopt during training.")
    parser.add_argument('--pretraining_model', default=None, type=str, 
                        help="Path to a local checkpoint file from which to load network weights.")

    # Network-specific parameters
    # ResNet (RN)
    parser.add_argument('--rn_arch', default='resnet50', type=str, 
                        help="The starting ResNet architecture. Accepted values: resnet50, resnet101, resnet152.")
    parser.add_argument('--rn_head', default=[2048,1000], type=int, nargs="+", 
                        help="The width of the linear layers in the network classification head.")
    parser.add_argument('--rn_pretrained', default=None, type=str, help="Pretrain network with default weights. Can be either 'ImageNet' or 'RSP'/'MillionAid'.")
    parser.add_argument('--rn_first_trainable', default=0, type=int, 
                        help="Number of the first layer to train. Number associations in the network source code.")
    # SwinT (ST)
    parser.add_argument('--st_head', default=[2048,1000], type=int, nargs="+", 
                        help="The width of the linear layers in the network classification head.")
    parser.add_argument('--st_pretrained', default=None, type=str, help="Pretrain network with default weights. Can be either 'ImageNet' or 'RSP'/'MillionAid'.")
    parser.add_argument('--st_first_trainable', default=0, type=int, 
                        help="Number of the first layer to train. Number associations in the network source code.")
    # Optimizer
    parser.add_argument('--optimizer', default='sgd', type=str, help="The optimizer to use during training")
    # Optimizer-specific parameters
    parser.add_argument('--sgd_momentum', default=0, type=float, help="The momentum parameter for SGD optimizer.")
    parser.add_argument('--sgd_weight_decay', default=0, type=float, help="The weight decay parameter for SGD optimizer.")

    # Early stopping
    parser.add_argument('--es_patience', default=100, type=int, help="The patience for early stopping during training.")
    parser.add_argument('--es_min_delta', default=0.1, type=float, help="The minimum delta to consider for early stopping.")
    
    # DATA AUGMENTATION
    # TRAINING TIME
    parser.add_argument('--train_resize', default=None, type=int, nargs="+", 
                        help="The size for resizing images in validation set at training time. Format assumed to be 'width height'")
    parser.add_argument('--train_fliph_prob', default=.0, type=float, help="Probability of flipping an image horizontally for augmentation.")
    parser.add_argument('--train_flipv_prob', default=.0, type=float, help="Probability of flipping an image vertically for augmentation.")
    parser.add_argument('--train_rotate90s', default=False, action='store_true', help="Defines whether to rotate the image of multiples of 90 degrees for augmentation.")
    parser.add_argument('--train_pad', default=None, type=int, help="The amount of pixels to pad the image on each side during augmentation.")
    # VALIDATION TIME
    # Same as above, but applicable at validation time
    parser.add_argument('--val_resize', default=None, type=int, nargs="+", 
                        help="The size for resizing images in validation set at training time. Format assumed to be 'width height'")
    parser.add_argument('--val_pad', default=None, type=int, help="The amount of pixels to pad the image on each side during augmentation.")

    # INFERENCE TIME
    parser.add_argument('--infer_resize', default=None, type=int, nargs="+", 
                        help="The size for resizing images at inference time. Format assumed to be 'width height'")
    parser.add_argument('--infer_pad', default=None, type=int, help="The amount of pixels to pad the image on each side during augmentation.")

    # Output Paths
    parser.add_argument('--out_log_file', default='result/train.log', type=str)
    parser.add_argument('--out_checkpoint_dir', default='result/checkpoints', type=str)
    parser.add_argument('--out_cam_dir', default='result/cams', type=str)
    parser.add_argument('--out_pred_dir', default='result/predictions', type=str)
    parser.add_argument('--out_tb_dir', default='result/tb_logs', type=str)

    ## Step
    parser.add_argument('--make_train', default=False, action='store_true')
    parser.add_argument('--make_pred', default=False, action='store_true')
    parser.add_argument('--make_cam', default=False, action='store_true')

    args = parser.parse_args()

    # Sanity check on log file name
    assert args.out_log_file.endswith('.log'), "Invalid name for the output log file."

    # Set seed passed as argument
    seed = args.experiment_seed
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    # Create output folders if non-existent
    os.makedirs(args.out_checkpoint_dir, exist_ok=True)
    os.makedirs(args.out_cam_dir, exist_ok=True)
    os.makedirs(args.out_pred_dir, exist_ok=True)
    os.makedirs(args.out_tb_dir, exist_ok=True)

    # Initialize logging
    logger = Logger(args.out_log_file)
    logger.write(f'{pformat(vars(args))}\n\n')  # writes a dictionary with the values of all parsed arguments

    # Execute training, if required
    if args.make_train is True:
        train.run(args)

    # Execute inference (enfolds computing predictions and extracting CAMs)
    if args.make_cam is True or args.make_pred is True:
        infer.run(args)