import os
import numpy as np

'''
note format: (x,y,w,h)

these are relative to the entire img
==> if resize pixel, this still going to work!!
'''


path='C:/Users/USER/PycharmProjects/reg_compitition/experiment_3/data/yolo'
names=os.listdir(path)

#change 15 to 0
for name in names:
    with open(f'{path}/{name}') as f:
        texts=f.readlines()
        for idx,text in enumerate(texts):
            text=text.replace('15','0')
            texts[idx]=text
    with open(f'{path}/{name}','w') as f:
        f.writelines(texts)