import torch
import torch.nn as nn

architecture_config=[
    (7,64,2,3), #112
    'M', #56
    (3,192,1,1),#56
    'M', #28
    (1,128,1,0), #28
    (3,256,1,1), #28
    [(1,256,1,0),(3,512,1,1),2],
    (1,1024,1,0), #28
    (3,1024,2,0), #13
]

#from SENet Block Structure
class Channel_Attention(nn.Module):
    def __init__(self,in_channel,ratio=8): #note: in_channel>=ratio
        super(Channel_Attention, self).__init__()
        self.mlp=nn.Sequential(
            nn.Linear(in_channel,in_channel//ratio,bias=False),
            nn.ReLU(),
            nn.Linear(in_channel//ratio,in_channel,bias=False)
        )

        self.global_avg=nn.AdaptiveAvgPool2d(1)
        self.global_max=nn.AdaptiveAvgPool2d(1)
        self.sigmoid=nn.Sigmoid()
        self.in_channel=in_channel

    def forward(self, x):
        avg_=self.mlp(self.global_avg(x).permute(0,2,3,1)) #(1,1,1,c)
        max_=self.mlp(self.global_max(x).permute(0,2,3,1)) #(1,1,1,c)
        channel_attention=self.sigmoid(avg_+max_).permute(0,3,1,2) #(1,c,1,1)
        return channel_attention*x #(1,c,h,w)


class Spatial_Attention(nn.Module):
    def __init__(self,out_channels,kernel_size=7):
        super(Spatial_Attention, self).__init__()
        same_padding_size=(kernel_size-1)//2
        self.conv=nn.Conv2d(kernel_size=kernel_size,
                            in_channels=2,
                            out_channels=out_channels,
                            padding=same_padding_size)
        self.sigmoid=nn.Sigmoid()
    def forward(self, x):
        avg_=torch.mean(x,dim=1,keepdim=True) #(1,1,h,w)
        max_,_=torch.max(x,dim=1,keepdim=True) #(1,1,h,w)
        concat=torch.cat([max_,avg_],dim=1) #(1,2,h,w)
        spatial_attention=self.sigmoid(self.conv(concat)) #(1,1,h,w)
        return spatial_attention*x #(1,c,h,w)




class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,ratio=8,attention=True,**kwargs):
        super(ConvBlock,self).__init__()
        self.conv=nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            bias=False,
                            **kwargs)
        self.channel_attention=Channel_Attention(in_channel=out_channels,ratio=ratio)
        self.spatial_attention=Spatial_Attention(out_channels=1)

        self.batchnorm=nn.BatchNorm2d(out_channels)
        self.leakyrelu=nn.LeakyReLU(0.1,inplace=False)
        self.attention=attention
    def forward(self,x):
        x=self.conv(x)
        x=self.batchnorm(x)
        x=self.leakyrelu(x)
        if self.attention:
            x=self.channel_attention(x)
            x=self.spatial_attention(x)
        return x


class yolov1(nn.Module):
    def __init__(self,in_channels=3,**kwargs):
        super(yolov1,self).__init__()
        self.architecture=architecture_config
        self.in_channels=in_channels
        self.darknet=self.create_conv_layers(self.architecture)
        self.fc=self.create_fc(**kwargs)

    def forward(self,x):
        x=self.darknet(x)
        x=torch.flatten(x,start_dim=1) #batch之後都flatten
        x=self.fc(x)
        return x

    def create_conv_layers(self,architecture):
        layers=[]
        in_channels=self.in_channels
        for idx,x in enumerate(architecture):
            #Conv
            if type(x)==tuple:
                if idx==0:
                    layers.append(ConvBlock(in_channels=in_channels,
                                            out_channels=x[1],
                                            kernel_size=x[0],
                                            attention=False,
                                            ratio=1,
                                            stride=x[2],
                                            padding=x[3]))
                else:
                    layers.append(ConvBlock(in_channels=in_channels,
                                            out_channels=x[1],
                                            kernel_size=x[0],
                                            ratio=8,
                                            stride=x[2],
                                            padding=x[3]))
                in_channels=x[1]
            #Maxpooling
            elif type(x)==str:
                layers.append(nn.MaxPool2d(kernel_size=2,stride=2))

            #list:inception net like
            else:
                conv1=x[0]
                conv2=x[1]
                n_repeats=x[2]
                for i in range(n_repeats):
                    layers.append(ConvBlock(in_channels=in_channels,
                                            out_channels=conv1[1],
                                            kernel_size=conv1[0],
                                            ratio=8,
                                            stride=conv1[2],
                                            padding=conv1[3]))
                    layers.append(ConvBlock(in_channels=conv1[1],
                                            out_channels=conv2[1],
                                            kernel_size=conv2[0],
                                            ratio=8,
                                            stride=conv2[2],
                                            padding=conv2[3]))
                    in_channels=conv2[1]

        return nn.Sequential(*layers)

    def create_fc(self,split_size,num_boxes,num_classes):
        '''
        cells num on img:sxs
        num anchor boxes:num_boxes
        '''
        s,b,c=split_size,num_boxes,num_classes
        fcnn=nn.Sequential(
            nn.Linear(1024*s*s,496), #origin paper is 4960, 資料集少不需要這麼多
            nn.Dropout(0.4),
            nn.LeakyReLU(0.1,inplace=False),
            nn.Linear(496,s*s*(c+b*5)),
        )
        return fcnn
