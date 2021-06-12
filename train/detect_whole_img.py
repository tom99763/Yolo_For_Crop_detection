import torch
from attention_model import yolov1
from utils.non_max_suppression import nms,nms2
from torchvision.transforms import transforms,Compose
from utils.bboxes import cellboxes_to_boxes
from d2l import torch as d2l
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import joblib
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

#model output : (batch,s*s*(c+5*b))
b,c,s=2,1,13
window_size=224
load_model_file='../save/attention_model_v2.pt'

def bbox_to_rect(bbox, color):
    return d2l.plt.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0],
                             height=bbox[3] - bbox[1], fill=False,
                             edgecolor=color, linewidth=0.7)

def transform_upper_lower(boxes):
    upper_lower_bboxes=[]
    bboxes=[]
    for box in boxes:
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        lower_left=box[0] + box[2] / 2
        lower_right=box[1] + box[3] / 2
        bboxes.append(box)
        upper_lower_bboxes.append([upper_left_x,upper_left_y,lower_left,lower_right])
    return bboxes,upper_lower_bboxes

def detect(img,stride,model):
    ''':return
    img : (3000,2000) , (2304,1728)
    '''
    total_bboxes=[]
    transform = Compose([transforms.ToTensor(), transforms.Normalize([0, 0, 0], [1, 1, 1])])

    print('start detect...')
    for y in range(0,img.shape[0],stride):
        for x in range(0,img.shape[1],stride):
            if y+window_size>=img.shape[0] or x+window_size>=img.shape[1]:
                continue
            print('position:', y, x)
            window=img[y:y+window_size,x:x+window_size,:]
            h,w,_=window.shape
            window=transform(window).unsqueeze(0).to('cuda') #(1,3,448,448)
            model.eval()
            pred=model(window) #(1,s,s,c+5*b)
            bboxes = cellboxes_to_boxes(pred, S=s, b=b, c=c)
            bboxes = nms(bboxes[0], iou_threshold=0.05, prob_threshold=0.4, box_format="midpoint")
            bboxes=[[box[0],box[1],x+box[2]*w,y+box[3]*h,box[4]*w,box[5]*h] for box in bboxes]
            total_bboxes=total_bboxes+bboxes

    print('total_detected bboxes:',len(total_bboxes))
    print('suppress all boxes...')
    final_nms_bboxes=nms2(total_bboxes, iou_threshold=0.05, box_format="midpoint")
    print('transform upper lower...')
    final_bboxes ,upper_lower_bboxes=transform_upper_lower(final_nms_bboxes)
    print('done')

    return final_bboxes,upper_lower_bboxes



def main(img,stride):
    model=yolov1(num_boxes=b,num_classes=c,split_size=s).to('cuda')
    checkpoint=torch.load(load_model_file)
    model.load_state_dict(checkpoint["state_dict"])
    final_bboxes,upper_lower_bboxes=detect(img,stride,model)
    return final_bboxes,upper_lower_bboxes

if __name__=='__main__':
    names=os.listdir('../data/test_public')
    for name in names:
        print(name)
        name=name.replace('.JPG','')
        path = f'../data/test_public/{name}.JPG'
        img = d2l.plt.imread(path)
        stride = 50

        final_bboxes, upper_lower_bboxes = main(img, stride)

        plt.figure()
        d2l.set_figsize()
        fig = d2l.plt.imshow(img)
        for box in upper_lower_bboxes:
            fig.axes.add_patch(bbox_to_rect(box, 'red'))
        plt.savefig(f'../res/{name}_res.png')

        pickle = np.array(final_bboxes)[:, :2].astype('int32')
        df = pd.DataFrame(pickle)
        #df.to_csv(f'../submit/{name}.csv', header=False, index=False)
