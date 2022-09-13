import logging
import os
import numpy as np
import torch
import torch.utils.data
from .engine import evaluate
from torchvision.models.detection.ssdlite import SSDLiteHead
import torchvision
import torchvision.models.detection.ssdlite
import torch.nn as nn
from functools import partial
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from modeltrainer.constants import FASTERRCNN, MOBILE_NET
from .train_early_stop import train_model
from .datasets import AllDataset , collate_fn

logger = logging.getLogger(__name__)

def get_model_mobilenet(num_classes):
  # load an object detection model pre-trained on COCO
  model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained = True, trainable_backbone_layers=2)
  # get the number of input features for the classifier
  in_channels = det_utils.retrieve_out_channels(model.backbone, (320, 320))
  num_anchors = model.anchor_generator.num_anchors_per_location()
  norm_layer  = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
  # replace the pre-trained head with a new on
  model.head.classification_head = SSDLiteClassificationHead(in_channels, num_anchors, num_classes, norm_layer)
  return model

def get_model_faster_rcnn(num_classes):
  # load an object detection model pre-trained on COCO
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT', trainable_backbone_layers=2)
  # get the number of input features for the classifier
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  # replace the pre-trained head with a new on
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)
  return model

def load_existing_model_faster_rcnn():
  pass

def retrain_model(model_name, pretrained=True):
    try:
        global device, trainloader, testloader, validloader
        valid_size = 0.2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_path='training_data'
        filename='annotations_augmented_flip_h2.csv'

        trainset = AllDataset(data_path, os.path.join(data_path,filename), None)
        trainset_size = int(len(trainset) * 0.8)
        testset_size = len(trainset) - trainset_size
        trainset, testset = torch.utils.data.random_split(trainset, [trainset_size, testset_size])
        # testset = AllDataset(data_path, os.path.join(data_path,filename), None)
        
        num_train = len(trainset)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]
        # trainset_size = int(len(trainset) * 0.8)
        # validset_size = len(trainset) - trainset_size
        # trainset, validset = torch.utils.data.random_split(trainset, [trainset_size, validset_size])
        
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, sampler=train_sampler, num_workers=12, collate_fn=collate_fn, drop_last=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=4, num_workers=8, collate_fn=collate_fn)
        validloader = torch.utils.data.DataLoader(trainset, batch_size=4, sampler=valid_sampler, num_workers=8, collate_fn=collate_fn, drop_last=True)
        
        logger.info("Data loader successfully created !")

        if(model_name == MOBILE_NET):
            model = get_model_mobilenet(39)
            if not pretrained:
              existing_model_path = 'mobilenet_adamw.pt'
              model.load_state_dict(torch.load(existing_model_path))
        elif(model_name == FASTERRCNN):
            model = get_model_faster_rcnn(39)
        else:
            logger.info('Retrain failed! Unknown Model!')
            return
          
        model.to(device)
        
        logger.info("Model imported")

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,momentum=0.9, weight_decay=0.0005)
        # optimizer = torch.optim.AdamW(params,lr=0.0003,betas=(0.9,0.999),eps=1e-08,weight_decay=0.0005,amsgrad=False)
        
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)

        n_epochs = 1
        patience = 10
        batch_size = 4

        logger.info("Starting training ...")
        model = train_model(model, model_name, device, trainloader, validloader, optimizer, lr_scheduler, batch_size, patience, n_epochs)

        logger.info("Counting Precision and Recall...")
        evaluate(model, testloader, device=device)
        
        logger.info("All Done !!!")
    except Exception as e:
        raise e

