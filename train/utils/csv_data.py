import pandas as pd
import os

label_names=os.listdir('C:/Users/USER/PycharmProjects/reg_compitition/special_case/data/yolo')

imgs=[]

for name in label_names:
    img_name=name.replace('.txt','.jpg')
    imgs.append(img_name)

df=pd.DataFrame({'img':imgs,'label':label_names})
print(df)
df.to_csv('C:/Users/USER/PycharmProjects/reg_compitition/special_case/data/train.csv',index=False)

