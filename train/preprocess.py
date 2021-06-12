import torch
import os
import pandas as pd
from PIL import Image
'''
label裡面的x,y,w,h是相對於整張圖片的比例座標,所以圖片縮放也能work
'''


class Datasets(torch.utils.data.Dataset):
    def __init__(
            self, csv_file, img_dir, label_dir, S, B, C, transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        #做圖片處理 e.g. centercrop、randomcrop
        if self.transform:
            # image = self.transform(image)
            image, boxes = self.transform(image, boxes)

        # 轉成每個grid一個對應一個label
        #因為是label所以只有一個box,如果用2個boxes以上做預測,第一個box以後都填0
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            #處理座標

            ##label裡面的x,y,w,h從整張圖的比例座標轉成相對於一個grid的比例
            i, j = int(self.S * y), int(self.S * x) #box在第i列第j行的grid裡

            #e.g.假設分3x3個grids,原(x,y)=(0.3,0.3)
            #去小數點:(0.3*3=0.9,0.3*3=0.9)-->box在第0行第0個grid
            #相對於grid的比例座標:(0.9-0,0.9-0)-->這點對於第0個grid的x,y是(0.9,0.9)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            #w,h轉為相對於grid的w,h
            #e.g.假設分3x3個grids,相對於整張圖(w=0.3,h=0.5),
            #轉成(0.3*3,0.3*5)-->(0.9,1.5)
            #w,h--->grid寬的0.9倍,grid高的1.5倍
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            #類別跟機率項處理
            if label_matrix[i, j, self.C] == 0:
                # 機率p填1
                label_matrix[i, j, self.C] = 1

                #Box座標跟大小
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                label_matrix[i, j,self.C+1:self.C+5] = box_coordinates

                #data對應的類別index填1
                label_matrix[i, j, class_label] = 1

        return image, label_matrix


'''test dataset
csv_file='../data/train.csv'
img_dir='../data/img'
label_dir='../data/yolo'
ds=Datasets(csv_file,img_dir,label_dir,S=13,C=1,B=2)
img,label_matrix=ds.__getitem__(0)
print(label_matrix)
print(img)
'''







