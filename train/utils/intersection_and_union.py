import torch
'''
a bounding box contains info:(x1,y1,x2,y2) 
where (x1,y1) is top-left corner and (x2,y2) is down-right corner


evaluate how good the bounding box is
IOU(ground truth,prediction)=area of intersection/area of union
IOU>0.5 'decent'
IOU>0.7 'pretty good'
IOU> 'almost perfect'

intersection : 
     top-left corner (x',y'):  x'=max(x1_1,x1_2),y'=max(y1_1,y1_2)
     down-right corner (x_,y_) : x_=min(x2_1,x2_2) , y_=min(y2_1,y2_2)
union: A+B-A&B
'''

'''
box_format
  1. corners:(x1,y1,x2,y2) 左上右下
  2. midpoint: (x,y,w,h) 中心點和寬高--->要轉成corners的形式
     左上:(中心點x-寬/2, 中心點y-高/2)
     右下:(中心點x+寬/2, 中心點y+高/2)
'''



def iou(boxes_pred,boxes_true,box_format='midpoint'):
    '''
    boxes_pred:(batch,s,s,4) , 4-->x,y,w,h
    boxes_true:(batch,s,s,4) , 4-->x,y,w,h

    boxes_pred跟boxes_true算每個grid算iou,把x、y、w、h用掉後output大小是(batch,s,s)
    每個grid會得到一個iou分數,如果grid裡面沒box就是iou就是0
    '''

    if box_format=='midpoint':
        x,y,w,h=boxes_true[...,0],boxes_true[...,1],boxes_true[...,2],boxes_true[...,3]
        x_hat,y_hat,w_hat,h_hat=boxes_pred[...,0],boxes_pred[...,1],boxes_pred[...,2],boxes_pred[...,3]

        x1,y1=x-w/2,y-h/2
        x2,y2=x+w/2,y+h/2

        x1_hat,y1_hat=x_hat-w_hat/2,y_hat-h_hat/2
        x2_hat,y2_hat=x_hat+w_hat/2,y_hat+h_hat/2


    elif box_format=='corner':
        x1,y1,x2,y2=boxes_true[...,0],boxes_true[...,1],boxes_true[...,2],boxes_true[...,3]
        x1_hat, y1_hat, x2_hat, y2_hat = boxes_pred[...,0],boxes_pred[...,1],boxes_pred[...,2],boxes_pred[...,3]

    top_left_corner_x = torch.max(x1, x1_hat)
    top_left_corner_y = torch.max(y1, y1_hat)
    down_left_corner_x = torch.min(x2, x2_hat)
    down_left_corner_y = torch.min(y2, y2_hat)

    #如果沒有交集--->邊長<0 , 用0代替
    intersection=(down_left_corner_x-top_left_corner_x).clamp(0)* \
                 (down_left_corner_y - top_left_corner_y).clamp(0)

    box_hat_area=torch.abs((x1_hat-x2_hat)*(y1_hat-y2_hat))
    box_true_area=torch.abs((x1-x2)*(y1-y2))
    union=box_hat_area+box_true_area-intersection
    iou_score=intersection/(union+1e-6)

    return iou_score

'''
box_pred=torch.randn(5,7,7,4)
box_true=torch.randn(5,7,7,4)
print('batch_iou',iou(box_pred,box_true,box_format='midpoint').shape)
'''
