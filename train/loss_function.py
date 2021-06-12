import torch
import torch.nn as nn
from utils.intersection_and_union import iou

'''
loss : 
   if midpt or class:
      (x-x')^2+(y-y')^2
   if width and hieght:
      (sqrt(w)-sqrt(w'))^2+(sqrt(h)-sqrt(h))^2 , sqrt aim to deal with large bounding box


go through every cells:
    go through each bounding box:
        if there is object(have highest IOU than other box in the cell):
            loss(box midpt)  with lambda=lambda_coord
            loss(box width height)  with lambda=lambda_coord
            loss(class)  with lambda=1  去增加預測是object的機率
        else:
            loss(class) with lambda=lambda_noob 去降預測是object的機率
    
    if there is object:
        for c all class:
           (p(c)-p(c'))^2  #去增加每個class預測是object的機率
'''

class yolo_loss(nn.Module):
    def __init__(self,s=7,b=2,n_classes=20,lambda_coord=5,lambda_noobj=0.5):
        super(yolo_loss,self).__init__()
        self.squared_loss=nn.MSELoss(reduction='sum')
        self.lambda_noobj=lambda_noobj
        self.lambda_coord=lambda_coord
        self.s=s #grid size
        self.b=b #num boxes each cell
        self.n_classes=n_classes
    def forward(self,pred,target):
        '''
        pred:(batch_size,num_pred,s,s), s=cell_size, num_pred:[c1,...cn,p,x,y,w,h,p',x',w',h',....]
        target:(batch_size,5,s,s) ,5:(p,x,y,w,h)
        '''
        pred=torch.reshape(pred,(-1,self.s,self.s,self.n_classes+self.b*5))
        target_box=target[:,:,:,self.n_classes:]
        i=self.n_classes
        j=self.n_classes+5
        for _ in range(self.b):
            pred_box=pred[:,:,:,i+1:j]
            #iou : 中心點 box_format=midpt
            iou_score=iou(pred_box,target_box[:,:,:,1:],box_format='midpoint') #return :(batch,s,s)
            if _==0:
                ious=iou_score.unsqueeze(0)
            else:
                ious_=torch.cat([ious,iou_score.unsqueeze(0)],dim=0)
                ious=ious_
            i+=5
            j+=5
        #每個區塊選出iou最高的box
        #idx:某個box在第幾張圖某個區塊分最好
        iou_maxes,best_box_idx=torch.max(ious.squeeze(-1),dim=0) #return : val,idx

        ##identity_obj_i , 在各個cell裡面有沒有object(0 or 1)
        object_mask=target[:,:,:,self.n_classes].unsqueeze(-1)


        '''
        算座標的loss: x,y,w,h
        flatten留下座標項
        '''
        box_predictions=torch.zeros((target.shape[0],target.shape[1],target.shape[2],4)).to(target.device)
        for n in range(self.b):
            part=pred[...,self.n_classes+(n*5)+1:self.n_classes+(n+1)*5]
            batch,row_idx,col_idx=torch.where(best_box_idx==n)  #找第n個box iou得分最高的區塊
            idx_stack=torch.stack((batch,row_idx,col_idx)).T.tolist()
            #b=batch,r=row_id,c=col_id
            for idx in idx_stack:
                b,r,c=idx
                box_predictions[b, r, c, :] = part[b, r ,c, :]


        #算w,h的時候是取sqrt,所以要先對w,h做sqrt
        #這裡有一個小trick:sqrt對負數會出現複變數,所以先取絕對值然後做完sqrt後用sign函數還正負號
        #Note torch.sign :  if >0:1 , elif =0:0   elif <0:-1
        box_predictions[...,2:4]=torch.sign(box_predictions[...,2:4].clone())*torch.sqrt(torch.abs(box_predictions[...,2:4].clone())+10**(-6))
        box_predictions_=object_mask*box_predictions

        box_targets=object_mask*target[:,:,:,self.n_classes+1:self.n_classes+5]  #broadcast
        box_targets[...,2:4]=torch.sqrt(box_targets[...,2:4].clone())

        box_loss=self.squared_loss(
            torch.flatten(box_predictions_,end_dim=-2),
            torch.flatten(box_targets,end_dim=-2)
        )

        '''
        1.算有object的loss : confidence there's an object 留下機率項
        2.算no object的loss: mask變成非object是1,留下機率項
        '''
        no_object_loss=torch.tensor(0.0).to(target.device)
        box_predictions = torch.zeros((target.shape[0], target.shape[1], target.shape[2],1)).to(target.device)
        for n in range(self.b):
            part=pred[...,self.n_classes+(n*5)].unsqueeze(-1)
            batch,row_idx,col_idx=torch.where(best_box_idx==n)  #找第n個box iou得分最高的區塊
            idx_stack=torch.stack((batch,row_idx,col_idx)).T.tolist()
            #b=batch,r=row_id,c=col_id
            for idx in idx_stack:
                b,r,c=idx
                box_predictions[b, r, c,:] = part[b, r ,c,:]

            #no_object_loss
            no_object_loss+=self.squared_loss(
                torch.flatten((1-object_mask)*pred[...,self.n_classes+(n*5)].unsqueeze(-1),start_dim=1),
                torch.flatten((1-object_mask)*target[...,self.n_classes].unsqueeze(-1),start_dim=1)
            )
        object_loss=self.squared_loss(
            torch.flatten(object_mask*box_predictions,end_dim=-2),
            torch.flatten(object_mask*target[...,self.n_classes].unsqueeze(-1),end_dim=-2)
        )
        '''
        算類別的loss
        flatten留下類別項
        '''
        class_loss=self.squared_loss(
            torch.flatten(object_mask*pred[...,:self.n_classes],end_dim=-2),
            torch.flatten(object_mask*target[...,:self.n_classes],end_dim=-2)
        )

        loss=self.lambda_coord*box_loss+1*object_loss+self.lambda_noobj*no_object_loss+class_loss

        return loss




'''label跟預測隨便生的,所以會output nan
n_classes=20
num_boxes = 2
n_channels = 5 * num_boxes + n_classes
num_regions = 7
batch_size = 3
loss = yolo_loss(s=num_regions, b=num_boxes, n_classes=n_classes)
y = Variable(torch.randn(batch_size, num_regions, num_regions, 5 * 1 + n_classes))  # ground truth
y.requires_grad = True
y_hat = torch.randn(batch_size, n_channels * num_regions * num_regions)  # prediction
loss_ = loss(y_hat, y)
print(loss_)
loss_.backward()
'''
