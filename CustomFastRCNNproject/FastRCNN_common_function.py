
import numpy as np
import os
import torch
import cv2 as cv
from PIL import Image
from torch.utils.data import Dataset

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

def open_camera():
    global image_raw
    global image_box
    cap = cv.VideoCapture(0)
    while (cap.isOpened()):
        ret, frame = cap.read()
        # frame = model_evalmode(frame)
        cv.imshow('image',frame)   
        key = cv.waitKey(1/60)
        # ESC
        if key == 27:
            break
    cap.release()
    cv.destroyAllWindows()