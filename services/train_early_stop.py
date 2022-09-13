import logging
from modeltrainer.constants import FASTERRCNN, MOBILE_NET
from .tools import EarlyStopping
import numpy as np
import torch
from .eval import eval_forward_mobilenet , eval_forward_faster_rcnn

logger = logging.getLogger("django")

def train_model(model, model_name, device, trainloader, validloader, optimizer, lr_scheduler, batch_size, patience, n_epochs):
    
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    if (model_name == MOBILE_NET):
        checkpoint_path = 'mobilenet_adamw.pt'
    elif (model_name == FASTERRCNN):
        checkpoint_path = 'faster_rcnn_adam.pt'
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True, path = checkpoint_path)
    
    for epoch in range(1, n_epochs + 1):
        logger.info(f'Epoch: {epoch}')
        ###################
        # train the model #
        ###################
	    # prep model for training
        model.train()
        for images, targets in trainloader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(images,targets)
            # calculate the loss
            # loss_dict = model(output, target)
            # print(batch)
            # print(output)
            losses = sum(loss for loss in output.values())
            # backward pass: compute gradient of the loss with respect to model parameters
            losses.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(losses.item())
        
        lr_scheduler.step()
        ######################    
        # validate the model #
        ######################
        print("validating data ...")
        model.eval()
        for images, targets in validloader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # forward pass: compute predicted outputs by passing inputs to the model
            if(model_name == MOBILE_NET):
                output = eval_forward_mobilenet(model, images,targets)
            elif (model_name == FASTERRCNN):
                output = eval_forward_faster_rcnn(model, images,targets)
            else:
                return
            # calculate the loss
            losses = sum(loss for loss in output[0].values())
            # record validation loss
            valid_losses.append(losses.item())

        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        epoch_len = len(str(n_epochs))
        
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        
        print(print_msg)
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(checkpoint_path))
    return  model, avg_train_losses, avg_valid_losses
