import torch
from utils.metrics_for_object import mAP
from loss_function import yolo_loss
from attention_model import yolov1
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from preprocess import Datasets
from tqdm import tqdm
from utils.bboxes import get_bboxes
import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    #optimizer.load_state_dict(checkpoint["optimizer"])


seed=123
torch.manual_seed(seed)

# parameters
batch_size=8
learning_rate=1e-3
step=25
device='cuda' if torch.cuda.is_available() else 'cpu'
weight_decay=0
epochs=1000
num_workers=2
pin_memory=False
load_model=True
shuffle=True
load_model_file= '../save/attention_model_v2.pt'
img_dir= '../data/train'
label_dir= '../data/yolo'

class compose(object):
    def __init__(self,transforms):
        self.transforms=transforms

    def __call__(self, img,bboxes):
        for t in self.transforms:
            img,bboxes=t(img),bboxes
        return img,bboxes

transform=compose([transforms.ToTensor(),transforms.Normalize([0,0,0],[1,1,1])])

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(train_loader,model,scheduler,optimizer,loss_fn):
    loop=tqdm(train_loader,leave=True)
    mean_loss=[]
    for idx,(x,y) in enumerate(loop):

        x,y=x.to(device),y.to(device)
        out=model(x)
        loss=loss_fn(out,y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #update progress bar
        loop.set_postfix(loss=loss.item())

        del x,y,loss,out
        torch.cuda.empty_cache()
    scheduler.step()

    print(f'mean loss:{sum(mean_loss)/len(mean_loss)}, lr:{get_lr(optimizer)}')

def main():
    torch.autograd.set_detect_anomaly(True)
    s=13
    b=2
    c=1

    #model_v1
    print('call model...')
    model=yolov1(num_boxes=b,num_classes=c,split_size=s).to(device)
    optimizer=optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    scheduler =optim.lr_scheduler.StepLR(optimizer,step_size=step,gamma=0.1)
    loss_fn=yolo_loss(s=s,b=b,n_classes=c,lambda_noobj=0.5)

    if load_model:
        print('load checkpoint....')
        load_checkpoint(torch.load(load_model_file),model,optimizer)

    print('get data...')
    train_csv='../data/train.csv'

    train_dataset=Datasets(train_csv,img_dir,label_dir,S=s,B=b,C=c,transform=transform)

    train_loader=DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    print('start training...')
    torch.cuda.empty_cache()
    for epoch in range(epochs):

        train(train_loader, model,scheduler,optimizer,loss_fn)

        print('evaluate...')
        #compute mAP
        pred_boxes,target_boxes=get_bboxes(train_loader,model,S=s,c=c,b=b,iou_threshold=0.5,threshold=0.5)
        mean_average_precision=mAP(pred_boxes,target_boxes,iou_threshold=0.6,num_classes=c)

        print(f'train mAP:{mean_average_precision}')

        if mean_average_precision>0.8:
            checkpoint={
                'state_dict':model.state_dict(),
                'optimizer':optimizer.state_dict()
            }
            save_checkpoint(checkpoint,filename=load_model_file)

if __name__=='__main__':
    main()


