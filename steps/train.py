import os
import time
import torch
import torch.nn as nn
import torcheval.metrics as metrics
import torchvision.transforms.v2 as transforms
# Specific imports
from tqdm import tqdm
# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from datasets import AWClassificationDataset
from misc.transforms import Rotate90s
from misc.trainutils import create_optimizer, create_model

def run(args):

    # Create transform lists for augmentation
    train_transform_list = []
    val_transform_list = []
    # Train set
    if args.train_fliph_prob > .0:
        train_transform_list.append(transforms.RandomHorizontalFlip(args.train_fliph_prob))
    if args.train_flipv_prob > .0:
        train_transform_list.append(transforms.RandomVerticalFlip(args.train_flipv_prob))
    if args.train_rotate90s:
        train_transform_list.append(Rotate90s())
    if args.train_resize is not None:
        train_transform_list.append(transforms.Resize(size=tuple(args.train_resize)))
    if args.train_pad is not None:
        train_transform_list.append(transforms.Pad(args.train_pad, fill=0, padding_mode='constant'))
    train_transform_list.append(transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))) # ImageNet normalization values
    # Validation set
    if args.val_resize is not None:
        val_transform_list.append(transforms.Resize(size=tuple(args.val_resize)))
    if args.val_pad is not None:
        val_transform_list.append(transforms.Pad(args.val_pad, fill=0, padding_mode='constant'))
    val_transform_list.append(transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))) # ImageNet normalization values

    # Create datasets for training & validation
    training_set = AWClassificationDataset(args.train_desc_file, args.train_image_folder, transform=transforms.Compose(train_transform_list))
    validation_set = AWClassificationDataset(args.val_desc_file, args.val_image_folder, transform=transforms.Compose(val_transform_list))

    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)

    # Create model instance
    model = create_model(args)

    # Move model to selected GPUs
    # Assumes that training will always be launched on GPU
    if len(args.gpus) > 1:
        device = 'cuda'
    else:
        device = f'cuda:{args.gpus[0]}'
    model = nn.parallel.DataParallel(model, args.gpus).to(device)

    # Instantiate loss function
    loss_fn = torch.nn.BCELoss()

    # Instantiate optimizer
    optimizer = create_optimizer(model.parameters(), args)
    # Instantiate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    # Initialize TensorBoard writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(os.path.join(args.out_tb_dir, f'run_{timestamp}'))

    best_vloss = 1_000_000.
    es_patience = 0  # Early stopping patience
    es_loss = 1_000_000. # Best validation loss used for early stopping (allows cumulative delta)

    # Save starting timestamp
    start_time= time.perf_counter()

    for epoch in range(1, args.num_epochs+1):

        print('EPOCH {}:'.format(epoch))
        
        # Make sure gradient tracking is on, before doing a pass over the data
        model.train()
        
        ### TRAINING FOR CURRENT EPOCH
        running_loss = 0.
        last_loss = 0.
        
        report_freq = int(len(training_set.images)/args.batch_size/15) ## TO REVISE

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.type(torch.FloatTensor).to(device)
            
            # Zero your gradients for every batch!
            optimizer.zero_grad()
            
            # Make predictions for this batch
            outputs = model(inputs)
            
            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()
            
            # Adjust learning weights
            optimizer.step()
            
            # Gather data and report
            running_loss += loss.item()
            if i % report_freq == report_freq-1:
                last_loss = running_loss / report_freq # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch * len(training_loader) + i + 1     # Compute iteration number for plotting on TensorBoard 
                writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.
                
        # Average loss in training
        avg_tloss = last_loss

        ### VALIDATION FOR CURRENT EPOCH

        # Initialize validation loss
        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population statistics for batch normalization.
        model.eval()

        # Define metrics for quantitative evaluation on the validation set
        accuracy = metrics.BinaryAccuracy()
        recall = metrics.BinaryRecall()
        precision = metrics.BinaryPrecision()
        f1 = metrics.BinaryF1Score()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(tqdm(validation_loader)):
                vinputs, vlabels = vdata
                vinputs, vlabels = vinputs.to(device), vlabels.type(torch.FloatTensor).to(device)
                voutputs = model(vinputs)
                # Update metrics
                outs_resh = torch.reshape(voutputs, shape=[voutputs.size()[0]])
                labs_resh = torch.reshape(vlabels, shape=[vlabels.size()[0]]).type(torch.IntTensor)
                accuracy.update(outs_resh, labs_resh)
                recall.update(outs_resh, labs_resh)
                precision.update(outs_resh, labs_resh)
                f1.update(outs_resh, labs_resh)
                # Compute and update validation loss
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_tloss, avg_vloss))
        # Print computed metrics
        print(f"Accuracy: {accuracy.compute()*100:.2f}")
        print(f"Recall: {recall.compute()*100:.2f}")
        print(f"Precision: {precision.compute()*100:.2f}")
        print(f"F1: {f1.compute()*100:.2f}")

        # Log the running loss averaged per batch for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           { 'Training' : avg_tloss, 'Validation' : avg_vloss },
                           epoch)
        # Log the validation metrics
        writer.add_scalars('Validation Metrics',
                           {'Accuracy': accuracy.compute(), 
                            'Precision': precision.compute(),
                            'Recall': recall.compute(),
                            'F1-Score': f1.compute()},
                            epoch)
        # Log the learning rate value
        writer.add_scalars('Learning rate', { 'LR' : scheduler.get_last_lr()[0]}, epoch)

        writer.flush()

        # Reset metrics for following epoch
        for metric in [accuracy, recall, precision, f1]:
            metric.reset()

        # Track best performance, save the model state, check for early stopping
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = os.path.join(args.out_checkpoint_dir, 'checkpoint.pth')
            torch.save(model.module.state_dict(), model_path)
            # Check to reset early stopping patience
            if (es_loss - avg_vloss) > args.es_min_delta:
                es_patience = -1 # Later common line updates to 0 in this case
                es_loss = avg_vloss # Update loss for early stopping
            
        # Update early stopping
        es_patience += 1
        print(f"Early stopping patience: {es_patience}")
        if es_patience == args.es_patience:
            break 

        # Update scheduler with latest validation loss
        scheduler.step(avg_vloss)

        # Save last-epoch model
        model_path = os.path.join(args.out_checkpoint_dir, 'last_ep.pth')
        torch.save(model.module.state_dict(), model_path)

        print("\n") # Add an empty line to increase readability

    # Save ending timestamp, then pretty-print expired time
    end_time = time.perf_counter()
    training_time = int(end_time-start_time) # Discard sub-second accuracy
    hs = int(training_time/(60*60))
    mins = int((training_time/60)%60)
    secs = int(training_time) % 60
    print(f"Training completed in: {str(hs)+':' if hs > 0 else ''}{str(mins).zfill(2)+':' if mins > 0 or hs > 0 else ''}{secs:02d}")



    