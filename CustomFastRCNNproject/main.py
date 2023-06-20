# -*- coding: utf-8 -*-
"""
Created on 2023/05/29



@author: Shawn YH Lee
"""
import argparse
import numpy as np
import torch
import os
import threading
import cv2 as cv
from PIL import Image
from torchsummary import summary
import torchvision
from torchvision.models import mobilenet_v3_small
from torchvision.models import MobileNet_V3_Small_Weights
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.io.image import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from torchvision.ops import box_iou
"""----------module switch setting----------"""
tqdm_module = True #progress bar
torchsummary_module = True  #model Visual
argparse_module = True
load_model_weight = True
train_model  = False
"""----------module switch setting end----------"""

"""----------argparse module----------"""
if argparse_module:    
    parser = argparse.ArgumentParser(description = 'Object detection')
    parser.add_argument('--database_path',type=str,default='../RCNN_sample_project/PennFudanPed/',help='datapath')
    parser.add_argument('--images_path',type=str,default='../RCNN_sample_project/PennFudanPed/PNGImages/',help='data images')
    parser.add_argument('--mask_path',type=str,default='../RCNN_sample_project/PennFudanPed/PedMasks/',help='data images mask')
    parser.add_argument('--modelpath',type=str,default='./model/',help='output model save path')
    parser.add_argument('--numpy_data_path',type=str,default='./numpydata/',help='output numpy data')
    parser.add_argument('--training_data_path',type=str,default='./training_process_data/',help='output training data path')

    parser.add_argument('--image_size',type=int,default= 400,help='image size')
    parser.add_argument('--num_classes',type=int,default= 2,help='num classes')
    parser.add_argument('--batch_size',type=int,default= 8,help='batch_size')
    parser.add_argument('--num_epoch',type=int,default= 100,help='num_epoch')
    parser.add_argument('--model',type= str,default='Fastrcnn_mobilenetV3small',help='model')
    parser.add_argument('--optimizer',type= str,default='Adam',help='optimizer')
    parser.add_argument('--loss',type= str,default='CrossEntropyLoss',help='Loss')
    parser.add_argument('--lr',type= float,default=1e-3,help='learningrate')
    args = parser.parse_args()
"""----------argparse module end----------"""
def mask_resize(masks_np,shape=(400,400)):
    masks_rs = np.zeros(shape,np.uint8)

    masks_np_center = (int(masks_np.shape[0]/2),int(masks_np.shape[1]/2))
    masks_rs_center = (int(shape[0]/2),int(shape[1]/2)) #resize shape

    for i in range(masks_rs.shape[0]):
        for j in range(masks_rs.shape[1]):
            masks_rs[i][j] = masks_np[int(i*masks_np_center[0]/masks_rs_center[0])]\
            [int(j*masks_np_center[1]/masks_rs_center[1])]
            
    return masks_rs

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
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        #resize the same size
        img = img.resize((args.image_size,args.image_size))
        mask = mask_resize(mask, (args.image_size,args.image_size))

        # if idx==62:
        #    plt.imshow(mask)
        #    plt.show()
        
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        # split the color-encoded mask into a set
        # of binary masks
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

        # ckeck whether is image resize well
        # print(target['boxes'])
        # box = draw_bounding_boxes(torch.tensor(img*255,dtype=torch.uint8), boxes=target['boxes'],width=4)
        # box = toPIL_fn(box)
        # box.show()

        return img, target

    def __len__(self):
        return len(self.imgs)
    
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

""" camera use
def model_evalmode(frame):
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).features
    backbone.out_channels = 576
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)
    model = FasterRCNN(backbone,
                   num_classes=100,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler,).to(device)
    weights = MobileNet_V3_Small_Weights.DEFAULT
    preprocess = weights.transforms()

    frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    img = to_pil_image(frame)
    batch = [preprocess(img).to(device)]
    model.eval()
    predictions = model(batch)[0]
    labels = [weights.meta["categories"][i] for i in predictions["labels"]]
    image_raw = torch.tensor(batch[0],dtype=torch.uint8)
    box = draw_bounding_boxes(image_raw, boxes=predictions["boxes"],
                          labels=labels,
                          colors="red",
                          width=4, font_size=30)
    im = to_pil_image(box.detach())
    im =np.array(im)
    im = cv.cvtColor(im,cv.COLOR_RGB2BGR)
    print(im.shape)
    return im
"""

train_transform = transforms.Compose([
    transforms.ToTensor(), 
    # transforms.Resize((args.image_size,args.image_size)),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]),#做正規化[-1~1]之間
])
label_transform = transforms.Compose([
    transforms.Resize((args.image_size,args.image_size)),
])
toPIL_fn =transforms.ToPILImage()


device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).features
backbone.out_channels = 576
if torchsummary_module:
    summary(backbone,input_size=(3, 400, 400),device='cpu')
anchor_generator = AnchorGenerator(sizes=((32, 64, 128),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)


model = FasterRCNN(backbone,num_classes=2,rpn_anchor_generator=anchor_generator,box_roi_pool=roi_pooler\
                   ,min_size=400,max_size=400,image_mean=[0.5,0.5,0.5],image_std=[0.5,0.5,0.5]\
                    ,rpn_pre_nms_top_n_train=1000,rpn_pre_nms_top_n_test=1000\
                        ,rpn_post_nms_top_n_train=50,rpn_post_nms_top_n_test=50).to(device)
if load_model_weight:
    model.load_state_dict(torch.load('./model/Fastrcnn_mobilenetV3small.pth'))
print(model)

train_set = PennFudanDataset(args.database_path,train_transform)
train_loader = DataLoader(dataset = train_set,batch_size = args.batch_size,shuffle=True,pin_memory=True,collate_fn=collate_fn)
test_loader = DataLoader(dataset = train_set,batch_size = args.batch_size,shuffle=False,pin_memory=True,collate_fn=collate_fn)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,amsgrad=False)
loss_value = []
mIOUs = []
# test 
# for images, targets in test_loader:
#     box = draw_bounding_boxes(torch.tensor(images[0]*255,dtype=torch.uint8), boxes=targets[0]['boxes'],width=4)
#     box = toPIL_fn(box)
#     box.show()
#     pass

    
if train_model:
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
        loss_value.append(total_loss)

            # model.eval()
            # predict = model(images.to(device))
        torch.save(model.state_dict(), args.modelpath +args.model +'.pth')
        # memory not enough
        # calculate IOU
        model.eval()
        with torch.no_grad():
            ious = []
            for images, targets in test_loader:
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                outputs = model(images.to(device), targets)
                for j in range(len(outputs)):
                    # process score >0.5
                    pred_boxes = outputs[j]['boxes']
                    target_boxes = targets[j]['boxes']
                    iou = box_iou(pred_boxes, target_boxes)
                    print(iou)
                    iou_mean = torch.mean(iou)
                    if torch.isnan(iou_mean):
                        continue
                    ious.append(iou_mean)
            avg_ious = sum(iou for iou in ious) / len(ious)
            mIOUs.append(avg_ious)
            print('avg_ious:', avg_ious)
        
        print(f"Epoch {epoch+1}/{args.num_epoch}, Loss: {total_loss/len(train_loader)}")
else:
    
    label = ['background','people']
    model.eval()
    img_path ="../RCNN_sample_project/PennFudanPed/PNGImages/FudanPed00006.png"

    """------------------model eval one shot--------------------"""
    img = Image.open(img_path)
    img = label_transform(img) # 3*400*400
    batch = [train_transform(img).to(device)]

    predictions = model(batch)[0]
    print(predictions)

    labels = [label[i] for i in predictions["labels"]]
    img_tensor = read_image(img_path)
    img_tensor = label_transform(img_tensor)

    box = draw_bounding_boxes(img_tensor, boxes=predictions["boxes"],
                            labels=labels,
                            #   colors="red",
                            width=1)

    im = to_pil_image(box.detach())
    im.show()

    """------------------model eval batch mIOU--------------------"""
    with torch.no_grad():
        ious = []
        for images, targets in test_loader:
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images.to(device), targets)

            for j in range(len(outputs)):
                pred_boxes = outputs[j]['boxes']
                # pred_scores = outputs[j]['scores']
                # scores_threshold = 0.8
                # pred_boxes = pred_boxes[pred_scores>=scores_threshold]

                target_boxes = targets[j]['boxes']
                print(pred_boxes, target_boxes)
                iou = box_iou(pred_boxes, target_boxes)
                iou_mean = torch.mean(iou)
                if torch.isnan(iou_mean):
                    continue
                ious.append(iou_mean)
        print(ious)
        avg_ious = sum(iou for iou in ious) / len(ious)
        mIOUs.append(avg_ious)
        print('avg_ious:', avg_ious)

