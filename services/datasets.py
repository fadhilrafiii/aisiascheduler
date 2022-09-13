import os
import torchvision.transforms as transforms
import torch
import torch.utils.data
from PIL import Image
import pandas as pd

def parse_one_annot(path_to_data_file, filename):
   data = pd.read_csv(path_to_data_file)
   boxes_array = data[data["filename"] == filename][["x_min", "y_min",        
   "x_max", "y_max"]].values
   
   return boxes_array

def get_label(path_to_data_file, filename):
   data = pd.read_csv(path_to_data_file)
   classes_array = data[data["filename"] == filename]["class"].values
   return classes_array

class AllDataset(torch.utils.data.Dataset):
  def __init__(self, root, data_file, transforms):
    self.root = root
    self.transforms = transforms
    self.imgs = sorted(os.listdir(os.path.join(root, "images")))
    self.path_to_data_file = data_file
  def __getitem__(self, idx):
    # load images and bounding boxes
    img_path = os.path.join(self.root, "images", self.imgs[idx])
    img = Image.open(img_path).convert("RGB")
    trans1 = transforms.ToTensor()
    # trans1 = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    images = trans1(img)
    box_list = parse_one_annot(self.path_to_data_file, self.imgs[idx])
    boxes = torch.as_tensor(box_list, dtype=torch.float32) 
    num_objs = len(box_list)
    # map each class class
    classes = get_label(self.path_to_data_file, self.imgs[idx])
    labels = torch.as_tensor(classes, dtype=torch.int64)
    # labels = torch.ones((num_objs,), dtype=torch.int64)
    image_id = torch.tensor([idx])
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:,0])
    # suppose all instances are not crowd
    iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
    target = {}
    target["boxes"] = boxes
    target["labels"] = labels
    target["image_id"] = image_id
    target["area"] = area
    target["iscrowd"] = iscrowd
    # if self.transforms is not None:
    #   img, target = self.transforms(img, target)
    return images, target
  def __len__(self):
    return len(self.imgs)

def collate_fn(batch):
    return tuple(zip(*batch))
