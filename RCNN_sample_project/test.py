
import os
import numpy as np
import torch
from PIL import Image

def mask_resize(masks_np):
    masks_rs = np.zeros((400,400),np.uint8)
    masks_np_center = (int(masks_np.shape[0]/2),int(masks_np.shape[1]/2))
    masks_rs_center = (int(400/2),int(400/2)) #resize shape
    print(masks_np_center,masks_rs_center)
    for i in range(masks_rs.shape[0]):
        for j in range(masks_rs.shape[1]):
            masks_rs[i][j] = masks_np[int(i*masks_np_center[0]/masks_rs_center[0])]\
            [int(j*masks_np_center[1]/masks_rs_center[1])]
    return masks_rs

def PILmask2RGB(path):
    masks = Image.open(path)
    masks_np = np.array(masks)

    # masks_np = mask_resize(masks_np)

    obj_ids = np.unique(masks_np)
    objs_color = np.zeros((len(obj_ids),3),dtype=np.uint8)
    for i in range(len(obj_ids)):
        objs_color[i][:] = np.random.randint(0,255,size=(3),dtype=np.uint8)
    objs_color[0][:] = [0,0,0] # set backbround = 0
    masks_RGB = np.zeros((masks_np.shape[0],masks_np.shape[1],3),dtype=np.uint8)
    for i in range(masks_np.shape[0]):
        for j in range(masks_np.shape[1]):
            obj_id = masks_np[i][j]
            masks_RGB[i][j][:] = objs_color[obj_id]
    masks_RGB = Image.fromarray(masks_RGB)
    return masks_RGB

def find_bounding_box(path):
    masks = Image.open(path)
    masks = np.array(masks)

    obj = np.unique(masks)
    obj_ids = obj[1:] #remove background
    masks = masks == obj_ids[:, None, None]
    boxes = []
    for i in range(len(obj_ids)):
        pos = np.where(masks[i])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])   
        boxes.append([xmin, ymin, xmax, ymax])
    return boxes
def count_area(boxes):
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    return area
def painting_boxes(path,boxes):
    image = Image.open(path)
    image_np = np.array(image)

    objs_color = np.zeros((len(boxes),3),dtype=np.uint8)
    for i in range(len(boxes)):
        objs_color[i] = np.random.randint(0,255,size=(3),dtype=np.uint8)
    print(objs_color)
    #paint rectangle box
    for i in range(len(boxes)):
        print(boxes[i])
        image_np[boxes[i][1]:boxes[i][3],boxes[i][0]] = objs_color[i]
        image_np[boxes[i][1]:boxes[i][3],boxes[i][2]] = objs_color[i]
        image_np[boxes[i][1],boxes[i][0]:boxes[i][2]] = objs_color[i]
        image_np[boxes[i][3],boxes[i][0]:boxes[i][2]] = objs_color[i]
    image = Image.fromarray(image_np)
    return image 

path = './PennFudanPed/PedMasks/FudanPed00004_mask.png'
path_raw = './PennFudanPed/PNGImages/FudanPed00004.png'
image = PILmask2RGB(path)
image.show()
boxes = find_bounding_box(path)
img = painting_boxes(path_raw,boxes)
img.show()
