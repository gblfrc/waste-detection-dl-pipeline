import numpy as np
import os
import torch
import torchvision.transforms.v2 as transforms
# Specific imports
from datasets import AWClassificationDatasetNamed
from tqdm import tqdm
from pytorch_grad_cam import GradCAM
from misc.inferutils import create_model, get_target_layers, get_reshape_transform

def run(args):

    # Create transform list
    infer_transform_list = []
    # Inference list
    if args.infer_resize is not None:
        infer_transform_list.append(transforms.Resize(size=tuple(args.infer_resize)))
    if args.infer_pad is not None:
        infer_transform_list.append(transforms.Pad(args.infer_pad, fill=0, padding_mode='constant'))
    infer_transform_list.append(transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))) # ImageNet normalization values

    # Create dataset for inference list
    infer_set = AWClassificationDatasetNamed(args.infer_list_file, args.infer_image_folder, transform=transforms.Compose(infer_transform_list))

    # Create data loader; do not shuffle
    infer_loader = torch.utils.data.DataLoader(infer_set, batch_size=args.batch_size, shuffle=False)

    # Update parameters for the inference phase
    args.pretraining_model = os.path.join(args.out_checkpoint_dir, 'checkpoint.pth')
    
    # Create model instance
    model = create_model(args)

    # Define layers and reshape transformation function to compute CAM
    target_layers = get_target_layers(model)
    reshape_transform = get_reshape_transform(model)

    # Assumes that inference will always be launched on GPU
    device = f'cuda:{args.gpus[0]}' # Always use a single GPU for limitation introduced by GradCam library which does not work properly with DP on multiple GPUs
    model = model.to(device)

    # Set model to evaluation mode
    model.eval();

    # Extract CAM (and predictions) if required
    if args.make_cam is True:
        # Construct the CAM object once, and then re-use it on many images
        cam = GradCAM(model=model, 
                      target_layers=target_layers,
                      reshape_transform=reshape_transform)
        # Iterate on batches in the DataLoader
        for input_tensor, _, names in tqdm(infer_loader, leave=False):
            # Actually compute CAM
            grayscale_cam = cam(input_tensor=input_tensor, targets=None) # Setting the targets to None implies computing CAMs for each output
            for i, name in enumerate(names):
                # Extract single CAM
                single_cam = np.asarray([grayscale_cam[i, :]], dtype=float) # Introduce custom format to support visualization and extension to multi-class tasks 
                # Save CAM
                out_cam_path = ''.join(os.path.join(args.out_cam_dir, name).split('.')[:-1]) + '.npy';
                np.save(out_cam_path, single_cam)
                # If computing predictions is required, save also predictions
                if args.make_pred is True:
                    out_pred_path = ''.join(os.path.join(args.out_pred_dir, name).split('.')[:-1]) + '.npy';
                    np.save(out_pred_path, cam.outputs[i,:].detach().cpu().numpy().astype(float))
    # Extract only predictions if required 
    elif args.make_pred is True:
        # Iterate on batches in the DataLoader
        for input_tensor, _, names in tqdm(infer_loader, leave=False):
            # Run model to extract predictions
            preds = model(input_tensor.to(device))
            for i, name in enumerate(names):
                out_pred_path = ''.join(os.path.join(args.out_pred_dir, name).split('.')[:-1]) + '.npy'; # This code is replicated from above. If changed above, change here as well. 
                np.save(out_pred_path, preds[i,:].detach().cpu().numpy().astype(float))