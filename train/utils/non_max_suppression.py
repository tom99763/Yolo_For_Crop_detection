import numpy as np
import torch
from utils.intersection_and_union import iou

'''Algorithm
把所有box讀進來,每個box:p,x,y,w,h
先去掉p沒過門檻的box
把剩下的box根據機率從高排到低

loop:
選第一個box,然後跟剩下的box check: 不同類就不去掉、iou重疊比門檻小不去掉
去掉第一個box,輪到第二個box,重複上一步
'''

def nms(bboxes,iou_threshold,prob_threshold,box_format='midpoint'):
    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > prob_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:

        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or iou(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms



def nms2(bboxes,iou_threshold,box_format='midpoint'):
    assert type(bboxes) == list

    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:

        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or iou(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]
        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


'''
boxes_pred = torch.tensor([[0, 0.8, 1, 1, 6, 3],
                           [0, 0.65, 2, 2, 5, 4],
                           [0, 0.7, 1, 3, 3, 4],
                           [1, 0.75, 1, 3, 3, 4],
                           [1, 0.9, 1, 3, 3, 5],
                           [0, 0.85, 1, 1, 6, 3],
                           [0, 0.9, 2, 2, 5, 3]])
final_boxes = nms(boxes_pred.tolist(), 0.5, 0.6)
print(final_boxes)
print('pred len:', len(final_boxes))
'''



