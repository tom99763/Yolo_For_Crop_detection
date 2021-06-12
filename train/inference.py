import torch
from attention_model import yolov1
from utils.non_max_suppression import nms
from torchvision.transforms import transforms,Compose
import cv2
from utils.bboxes import cellboxes_to_boxes
from d2l import torch as d2l
import matplotlib.pyplot as plt
import os
import joblib

os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

'''座標轉換
model output是相對一個grid的相對座標 , 返回到448x448圖的相對座標
如果是sliding window大圖的話就,再返回到整張大圖的座標
'''


#model output : (batch,s*s*(c+5*b))
s,b,c=13,2,1
load_model_file='../save/attention_model_v2.pt'

def bbox_to_rect(bbox, color):
    return d2l.plt.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0],
                             height=bbox[3] - bbox[1], fill=False,
                             edgecolor=color, linewidth=0.7)

def transform_upper_lower(boxes,h,w):
    bboxes=[]
    for box in boxes:
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        lower_left=box[0] + box[2] / 2
        lower_right=box[1] + box[3] / 2
        bboxes.append([upper_left_x*w,upper_left_y*h,lower_left*w,lower_right*h])
    return bboxes

def main(img):
    model=yolov1(num_boxes=b,num_classes=c,split_size=s).to('cuda')
    checkpoint=torch.load(load_model_file)
    model.load_state_dict(checkpoint["state_dict"])


    transform=Compose([transforms.ToTensor(),transforms.Normalize([0,0,0],[1,1,1])])
    img_=transform(img).unsqueeze(0).to('cuda')
    h,w,_=img.shape
    model.eval()
    pred=model(img_)
    bboxes = cellboxes_to_boxes(pred, S=s, b=b, c=c)
    #channels:[None,p,x,y,w,h]type:list
    bboxes = nms(bboxes[0], iou_threshold=0.2, prob_threshold=0.5)
    #channels:[x,y,w,h]
    bboxes=transform_upper_lower(bboxes,h,w)

    return bboxes

path='../data/train/IMG_170406_040409_0253_RGB3.JPG_7.jpg'
img = d2l.plt.imread(path)
bboxes=main(img)
d2l.set_figsize()
fig = d2l.plt.imshow(img)
for box in bboxes:
    fig.axes.add_patch(bbox_to_rect(box, 'red'))
plt.savefig('../res/result.png')
plt.show()
