import numpy as np
import torch
from collections import Counter
from utils.intersection_and_union import iou

'''
在object detection裡面,預測正確=iou(y,y')>theshold
true positive:正確的東西預測正確
false positive:沒有的東西你說有
recall:所有預測的box哪些是True Positive
precision:true positive/(true positive+false positive)
'''

#mean average precision
#step: run iou_threshold=0.5,0.55,...0.95,then average it
def mAP(
        pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):

    average_precisions = []

    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []


        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        amount_bboxes = Counter([gt[0] for gt in ground_truths])


        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros((val))


        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)


        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):

            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou_ = iou(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou_ > best_iou:
                    best_iou = iou_
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:

                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1


            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)

'''
boxes_pred=torch.tensor([[0,0,0.8,1,1,6,3],
                             [0,0,0.65,2,2,5,4],
                             [0,0,0.7,1,3,3,4],
                             [0,1,0.75,1,3,3,4],
                             [1,1,0.9,1,3,3,5],
                             [1,0,0.85,1,1,6,3],
                             [1,0,0.9,2,2,5,3]])
boxes_true = torch.tensor([[0, 0, 0.7, 1.5, 1.2, 5.4, 3.2],
                           [0, 0, 0.6, 2.5, 2.2, 5, 4.1],
                           [0, 1, 0.4, 1.1, 3.5, 3.2, 4.6],
                           [1, 1, 0.9, 1, 3.2, 3, 5],
                           [1, 0, 0.85, 1, 1.1, 6.5, 3]])
map_score = mAP(boxes_pred, boxes_true, iou_threshold=0.5, num_classes=2)
print(map_score)
'''