
# -*- coding: utf-8 -*-
"""
Created on 2023/05/15

Database is maskimage

@author: Shawn YH Lee
"""

"""----------import package----------"""
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision.models import mobilenet_v3_small
from torchvision.io.image import read_image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from torchvision.models.detection.rpn import AnchorGenerator
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms

"""----------import package end----------"""

"""----------module switch setting----------"""
tqdm_module = True #progress bar
torchsummary_module = True  #model Visual
argparse_module = True
load_model_weight = False
"""----------module switch setting end----------"""

"""----------argparse module----------"""
if argparse_module:    
    parser = argparse.ArgumentParser(description = 'Object detection')
    parser.add_argument('--database_path',type=str,default='./PennFudanPed/',help='datapath')
    parser.add_argument('--images_path',type=str,default='./PennFudanPed/PNGImages/',help='data images')
    parser.add_argument('--mask_path',type=str,default='./PennFudanPed/PedMasks/',help='data images mask')
    parser.add_argument('--modelpath',type=str,default='./model/',help='output model save path')
    parser.add_argument('--numpy_data_path',type=str,default='./numpydata/',help='output numpy data')
    parser.add_argument('--training_data_path',type=str,default='./training_process_data/',help='output training data path')

    parser.add_argument('--image_size',type=int,default= 400,help='image size')
    parser.add_argument('--num_classes',type=int,default= 2,help='num classes')
    parser.add_argument('--batch_size',type=int,default= 4,help='batch_size')
    parser.add_argument('--num_epoch',type=int,default= 100,help='num_epoch')
    parser.add_argument('--model',type= str,default='Fastrcnn_mobilenetV3small',help='model')
    parser.add_argument('--optimizer',type= str,default='Ranger',help='optimizer')
    parser.add_argument('--loss',type= str,default='CrossEntropyLoss',help='Loss')
    parser.add_argument('--lr',type= int,default=1e-3,help='learningrate')
    args = parser.parse_args()
"""----------argparse module end----------"""

"""----------function----------"""
class PennFudanDataset(Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])

        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        #resize th same size
        # resize_t = transforms.Resize((args.image_size,args.image_size))
        # mask = resize_t(mask)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # if idx==62:
        #    plt.imshow(mask)
        #    plt.show()
        
        # instances are encoded as different colors
        # example: return [0,1,2]
        obj_ids = np.unique(mask) 
        
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        # split the color-encoded mask into a set
        # of binary masks
        # example: return(2,N,M) [False,True]
        masks = mask == obj_ids[:, None, None]
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)
            #img, target = self.transforms(img, target)
        
        return img, target

    def __len__(self):
        return len(self.imgs)
train_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize((args.image_size,args.image_size)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]),#做正規化[-1~1]之間
])


def collate_fn(batch):
    images = []
    targets = []
    for img, target in batch:
        images.append(img)
        targets.append(target)
    # Resize images to the same size
    images = torch.stack(images)

    # Collect information from targets
    boxes = [target['boxes'] for target in targets]
    labels = [target['labels'] for target in targets]
    masks = [target['masks'] for target in targets]
    image_ids = [target['image_id'] for target in targets]
    areas = [target['area'] for target in targets]
    iscrowds = [target['iscrowd'] for target in targets]

    # Process boxes and resize them if needed
    # ...

    # Return the processed data
    processed_targets = []
    for target in targets:
        processed_target = {
            'boxes': target['boxes'],
            'labels': target['labels'],
            'masks': target['masks'],
            'image_id': target['image_id'],
            'area': target['area'],
            'iscrowd': target['iscrowd']
        }
        processed_targets.append(processed_target)

    return images, processed_targets
"""----------function end----------"""
train_set = PennFudanDataset(args.database_path,train_transform)
train_loader = DataLoader(dataset = train_set,batch_size = args.batch_size,shuffle=True,pin_memory=True,collate_fn=collate_fn)


model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
if load_model_weight:
    model.load_state_dict(torch.load('./model/Fastrcnn_mobilenetV3small.pth'))
num_classes = 2  # 1 class (person) + background
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
model.to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,betas=(0.6, 0.999))
loss = torch.nn.CrossEntropyLoss()

for epoch in range(args.num_epoch):
    total_loss = 0.0
    for images, targets in train_loader:
        model.train()
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # 模型的前向傳遞和計算損失
        loss_dict = model(images.to(device), targets)
        losses = sum(loss for loss in loss_dict.values())

        # 反向傳播和參數更新
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

        # model.eval()
        # predict = model(images.to(device))

    torch.save(model.state_dict(), args.modelpath +args.model +'.pth')
    print(f"Epoch {epoch+1}/{args.num_epoch}, Loss: {total_loss/len(train_loader)}")

"""   Sample code for using eval image
img = read_image("./PennFudanPed/PNGImages/FudanPed00001.png")#torchuint8(C,H,W)

# Step 1: Initialize model with the best available weights
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()
# Step 3: Apply inference preprocessing transforms
batch = [preprocess(img)]

# Step 4: Use the model and visualize the prediction
prediction = model(batch)[0]
labels = [weights.meta["categories"][i] for i in prediction["labels"]]
print(labels)
box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                          labels=labels,
                          colors="red",
                          width=4, font_size=30)

im = to_pil_image(box.detach())
im.show()

"""