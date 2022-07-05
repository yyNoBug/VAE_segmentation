import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math

def Normalization(norm_type, out_channels,num_group=1):
    if norm_type==1:
        return nn.InstanceNorm3d(out_channels)
    elif norm_type==2:
        return nn.BatchNorm3d(out_channels,momentum=0.1)
    elif norm_type==3:
        return GSNorm3d(out_channels,num_group=num_group)

class GSNorm3d(torch.nn.Module):
    def __init__(self, out_ch, num_group=1):
        super().__init__()
        self.out_ch = out_ch
        self.num_group=num_group
        #self.activation = nn.ReLU()
    def forward(self, x):
        interval = self.out_ch//self.num_group
        start_index = 0
        tensors = []
        for i in range(self.num_group):
            #dominator = torch.sum(x[:,start_index:start_index+interval,...],dim=1,keepdim=True)
            #dominator = dominator + (dominator<0.001)*1
            tensors.append(x[:,start_index:start_index+interval,...]/(torch.sum(x[:,start_index:start_index+interval,...],dim=1,keepdim=True)+0.0001))
            start_index = start_index+interval
        
        return torch.cat(tuple(tensors),dim=1)

class DoubleConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, norm_type=2,soft=False):
        super().__init__()
        activation = torch.nn.Softplus() if soft else torch.nn.ReLU(inplace=False)
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, out_ch, 3, padding=1),     
            Normalization(norm_type,out_ch),
            activation,
            torch.nn.Conv3d(out_ch, out_ch, 3, padding=1),
            Normalization(norm_type,out_ch),  
            activation,
            torch.nn.Conv3d(out_ch, out_ch, 3, padding=1),
            Normalization(norm_type,out_ch),  
            activation
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class DoubleConv_GS(torch.nn.Module):
    def __init__(self, in_ch, out_ch, num_group=1,soft=False):
        super().__init__()
        activation = torch.nn.Softplus() if soft else torch.nn.ReLU(inplace=False)
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, out_ch, 3, padding=1),     
            activation,
            torch.nn.Conv3d(out_ch, out_ch, 3, padding=1), 
            activation
        )
    def forward(self, x):
        x = self.conv(x)
        return x
class Up_GS(torch.nn.Module):
    def __init__(self, in_ch, out_ch,num_group=1,kernal_size=(2,2,2),stride=(2,2,2),soft=False):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='trilinear'),
            DoubleConv_GS(in_ch, out_ch, num_group,soft=False)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
class Down_GS(torch.nn.Module):
    def __init__(self, in_ch, out_ch, num_group=1,kernal_size=(2,2,2),stride=(2,2,2),soft=False):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, in_ch, kernal_size, stride=stride, padding=0),
            DoubleConv_GS(in_ch, out_ch, num_group,soft=False)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
class Conv_GS(torch.nn.Module):
    def __init__(self, in_ch, out_ch, num_group=1, activation=True, norm=True,soft=False):
        super().__init__()
        activation = torch.nn.Softplus() if soft else torch.nn.ReLU(inplace=True)
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, out_ch, 3, padding=1),
            activation
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, norm_type=2, num_group=1,activation=True, norm=True,soft=False):
        super().__init__()
        activation = torch.nn.Softplus() if soft else torch.nn.ReLU(inplace=True)
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, out_ch, 3, padding=1),
            Normalization(norm_type,out_ch),
            activation,
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class Up(torch.nn.Module):
    def __init__(self, in_ch, out_ch,norm_type=2,kernal_size=(2,2,2),stride=(2,2,2),soft=False):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(in_ch, in_ch, kernal_size, stride=stride, padding=0),
            DoubleConv(in_ch, out_ch, norm_type,soft=False)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Down(torch.nn.Module):
    def __init__(self, in_ch, out_ch, norm_type=2,kernal_size=(2,2,2),stride=(2,2,2),soft=False):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, in_ch, kernal_size, stride=stride, padding=0),
            DoubleConv(in_ch, out_ch, norm_type,soft=False)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

            

class GSConv3d(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, num_group=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, if_sub=None,trainable=True):
        super(GSConv3d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        #self.if_sub=torch.tensor(if_sub,dtype=torch.float).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        if not trainable:
            self.weight.requires_grad=False
        self.num_group = num_group
        self.interval = self.in_channels//self.num_group
    def forward(self, x):
        
        weight = torch.abs(self.weight)
        start_index = 0
        tensors = []
        for i in range(self.num_group):
            tensors.append(weight[:,start_index:start_index+self.interval,...]/torch.sum(weight[:,start_index:start_index+self.interval,...],1,keepdim=True))
            start_index += self.interval
        weight = torch.cat(tuple(tensors),1)
        return F.conv3d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class GSConvTranspose3d(nn.ConvTranspose3d):

    def __init__(self, in_channels, out_channels, kernel_size, num_group=1, stride=1,
                 padding=0, dilation=1, output_padding=0,groups=1, bias=False, if_sub=None,trainable=True):
        super(GSConvTranspose3d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, output_padding, groups,bias, dilation )
        #self.if_sub=torch.tensor(if_sub,dtype=torch.float).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        if not trainable:
            self.weight.requires_grad=False
        self.num_group = num_group
        self.interval = self.in_channels//self.num_group
    def forward(self, x, output_size=None):
        output_padding = self._output_padding(x, output_size, self.stride, self.padding, self.kernel_size, self.dilation)
        weight = torch.abs(self.weight)
        start_index = 0
        tensors = []
        for i in range(self.num_group):
            tensors.append(weight[:,start_index:start_index+self.interval,...]/torch.sum(weight[:,start_index:start_index+self.interval,...],1,keepdim=True))
            start_index += self.interval
        weight = torch.cat(tuple(tensors),1)
        return F.conv_transpose3d(x, weight, self.bias, self.stride,
                        self.padding,output_padding,  self.groups,self.dilation)
class SConv3d(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, if_sub=None,trainable=True):
        super(SConv3d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        #self.if_sub=torch.tensor(if_sub,dtype=torch.float).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        if not trainable:
            self.weight.requires_grad=False
    def forward(self, x):
        
        weight = self.weight
        weight_mean = weight.mean([2,3,4], keepdim=True)
        weight = weight - weight_mean

        return F.conv3d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
         
class VAE(torch.nn.Module):

    # [16,32,64,128,256,512]
    def __init__(self, n_channels, n_class, norm_type=2, n_fmaps=[8,16,32,64,128,256],dim=1024,soft=False):
        super().__init__()
        #self.SConv3d = SConv3d(1,n_fmaps[0],3,padding=1,bias=True)
        self.in_block = Conv(n_class, n_fmaps[0],norm_type=norm_type,soft=False)
        self.down1 = Down(n_fmaps[0], n_fmaps[1],norm_type=norm_type,soft=False)
        self.down2 = Down(n_fmaps[1], n_fmaps[2],norm_type=norm_type,soft=False)
        self.down3 = Down(n_fmaps[2], n_fmaps[3],norm_type=norm_type,soft=False)
        self.down4 = Down(n_fmaps[3], n_fmaps[4],norm_type=norm_type,soft=False)
        self.down5 = Down(n_fmaps[4], n_fmaps[5],norm_type=norm_type,soft=False)
        self.fc_mean = torch.nn.Linear(16384,dim)
        self.fc_std = torch.nn.Linear(16384,dim)
        self.fc2 = torch.nn.Linear(dim,16384)
        self.up1 = Up(n_fmaps[5],n_fmaps[4],norm_type=norm_type,soft=False)
        self.up2 = Up(n_fmaps[4],n_fmaps[3],norm_type=norm_type,soft=False)
        self.up3 = Up(n_fmaps[3],n_fmaps[2],norm_type=norm_type,soft=False)
        self.up4 = Up(n_fmaps[2],n_fmaps[1],norm_type=norm_type,soft=False)
        self.up5 = Up(n_fmaps[1],n_fmaps[0],norm_type=norm_type,soft=False)
        self.out_block = torch.nn.Conv3d(n_fmaps[0], n_class, 3, padding=1)
        self.final = nn.Softmax(dim=1)
        self.n_class = n_class
    def forward(self, x,if_random=False,scale=1,mid_input=False,dropout=0.0):
        #'pred_only','pred_recon',if_random=False
        #x = data_dict[in_key]
        # print(x.shape)
        
        if not mid_input:
            #input_res = data_dict.get(self.in_key2)
            #input_x = self.SConv3d(input_x)
            x = self.in_block(x)
            x = self.down1(x)
            x = self.down2(x)
            x = self.down3(x)
            x = self.down4(x)
            x = self.down5(x)
            x = x.view(x.size(0),16384)
            x_mean = self.fc_mean(x)
            x_std = nn.ReLU()(self.fc_std(x))
            #data_dict['mean'] = x_mean
            #data_dict['std'] = x_std
            z = torch.randn(x_mean.size(0),x_mean.size(1)).type(torch.cuda.FloatTensor)
            if if_random:
                x = self.fc2(x_mean+z*x_std*scale)
            else:
                x = self.fc2(x_mean)
        else:
            x = self.fc2(x)
        x = x.view(x.size(0),256,4,4,4)
        
        x = self.up1(x)
        if dropout: x = torch.nn.functional.dropout(x, p=dropout, training=True)
        x = self.up2(x)
        if dropout: x = torch.nn.functional.dropout(x, p=dropout, training=True)
        x = self.up3(x)
        if dropout: x = torch.nn.functional.dropout(x, p=dropout, training=True)
        x = self.up4(x)
        if dropout: x = torch.nn.functional.dropout(x, p=dropout, training=True)
        x = self.up5(x)
        if dropout: x = torch.nn.functional.dropout(x, p=dropout, training=True)
        x = self.out_block(x)
        x = self.final(x)
       
        #data_dict[out_key] = x
        if not mid_input:
            return x,x_mean,x_std
        else:
            return x
        
class Encoder(torch.nn.Module):
    # [16,32,64,128,256,512]
    def __init__(self, n_channels, dim, norm_type=2, n_fmaps=[8,16,32,64,128,256],soft=False):
        super().__init__()
        #self.SConv3d = SConv3d(1,n_fmaps[0],3,padding=1,bias=True)
        self.in_block = Conv(n_channels, n_fmaps[0],norm_type=norm_type,soft=False)
        self.down1 = Down(n_fmaps[0], n_fmaps[1],norm_type=norm_type,soft=False)
        self.down2 = Down(n_fmaps[1], n_fmaps[2],norm_type=norm_type,soft=False)
        self.down3 = Down(n_fmaps[2], n_fmaps[3],norm_type=norm_type,soft=False)
        self.down4 = Down(n_fmaps[3], n_fmaps[4],norm_type=norm_type,soft=False)
        self.down5 = Down(n_fmaps[4], n_fmaps[5],norm_type=norm_type,soft=False)
        self.fc1 = torch.nn.Linear(16384, 1024)
        self.fc2 = torch.nn.Linear(1024, 128)
        self.fc_mean = torch.nn.Linear(128, dim)
    def forward(self, x):
        #'pred_only','pred_recon',if_random=False
        #x = data_dict[in_key]
        # import pdb; pdb.set_trace()
        x = self.in_block(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = x.view(x.size(0),16384)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x_mean = self.fc_mean(x)
        x_mean = torch.sigmoid(x_mean)
        return x_mean

class Segmentation_GS(torch.nn.Module):

    # [16,32,64,128,256,512]
    def __init__(self, n_channels, n_class, norm_type=2, n_fmaps=[8,16,32,64,128,256]):
        super().__init__()
        #self.SConv3d = SConv3d(1,n_fmaps[0],3,padding=1,bias=True)
        self.in_block = Conv_GS(n_channels, n_fmaps[0],num_group=2,soft=False)
        self.down1 = Down_GS(n_fmaps[0], n_fmaps[1],num_group=2,soft=False)
        self.down2 = Down_GS(n_fmaps[1], n_fmaps[2],num_group=2,soft=False)
        self.down3 = Down_GS(n_fmaps[2], n_fmaps[3],num_group=4,soft=False)
        self.norm1 = GSNorm3d(n_fmaps[0],num_group=2)
        self.norm2 = GSNorm3d(n_fmaps[1],num_group=4)
        self.norm3 = GSNorm3d(n_fmaps[2],num_group=8)
        self.norm4 = GSNorm3d(n_fmaps[3],num_group=8)
        self.up2 = torch.nn.Upsample(scale_factor=2, mode='trilinear')
        self.up4 = torch.nn.Upsample(scale_factor=4, mode='trilinear')
        self.up8 = torch.nn.Upsample(scale_factor=8, mode='trilinear')

        self.out_block1 = Conv_GS(n_fmaps[0]+n_fmaps[1]+n_fmaps[2]+n_fmaps[3], 32 ,soft=False)
        self.out_block2 = torch.nn.Conv3d(32, n_class, 1, padding=0)
        self.final = nn.Softmax(dim=1)
        self.n_class = n_class
    def forward(self, data_dict,in_key,out_key):
        x = data_dict[in_key]
        #input_res = data_dict.get(self.in_key2)
        #input_x = self.SConv3d(input_x)
        x1 = self.in_block(x)
        x1_norm = self.norm1(x1)
        x2 = self.down1(x1)
        x2_norm = self.up2(self.norm2(x2))
        x3 = self.down2(x2)
        x3_norm = self.up4(self.norm3(x3))
        x4 = self.down3(x3)
        x4_norm = self.up8(self.norm4(x4))
        x = torch.cat((x1_norm,x2_norm,x3_norm,x4_norm),dim=1)
        x = self.out_block1(x)
        x = self.out_block2(x)
        x = self.final(x)
        data_dict[out_key] = x
        return data_dict

        
class Segmentation(torch.nn.Module):

    # [16,32,64,128,256,512]
    def __init__(self, n_channels, n_class, norm_type=2, n_fmaps=[8,16,32,64,128,256]):
        super().__init__()
        #self.SConv3d = SConv3d(1,n_fmaps[0],3,padding=1,bias=True)
        self.in_block = Conv(n_channels, n_fmaps[0],norm_type=norm_type,soft=False)
        self.down1 = Down(n_fmaps[0], n_fmaps[1],norm_type=norm_type,soft=False)
        self.down2 = Down(n_fmaps[1], n_fmaps[2],norm_type=norm_type,soft=False)
        self.down3 = Down(n_fmaps[2], n_fmaps[3],norm_type=norm_type,soft=False)
        self.down4 = Down(n_fmaps[3], n_fmaps[4],norm_type=norm_type,soft=False)


        self.up2 = Up(n_fmaps[4],n_fmaps[3],norm_type=norm_type,soft=False)
        self.up3 = Up(n_fmaps[3],n_fmaps[2],norm_type=norm_type,soft=False)
        self.up4 = Up(n_fmaps[2],n_fmaps[1],norm_type=norm_type,soft=False)
        self.up5 = Up(n_fmaps[1],n_fmaps[0],norm_type=norm_type,soft=False)
        self.out_block = torch.nn.Conv3d(n_fmaps[0], n_class, 3, padding=1)
        self.final = nn.Softmax(dim=1)
        self.n_class = n_class
    def forward(self, data_dict,in_key,out_key, dropout=0.0):
        x = data_dict[in_key]
        #input_res = data_dict.get(self.in_key2)
        #input_x = self.SConv3d(input_x)
        x1 = self.in_block(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up2(x5)
        if dropout: x = torch.nn.functional.dropout(x, p=dropout, training=True)
        x = self.up3(x)+x3
        if dropout: x = torch.nn.functional.dropout(x, p=dropout, training=True)
        x = self.up4(x)+x2
        if dropout: x = torch.nn.functional.dropout(x, p=dropout, training=True)
        x = self.up5(x)
        if dropout: x = torch.nn.functional.dropout(x, p=dropout, training=True)
        x = self.out_block(x)
        if dropout: x = torch.nn.functional.dropout(x, p=dropout, training=True)
        x = self.final(x)
        data_dict[out_key] = x
        return data_dict
        
class Fusion(torch.nn.Module):

    # [16,32,64,128,256,512]
    def __init__(self, n_channels_img, n_channels_mask, n_class, norm_type=2, n_fmaps=[8,16,32,64,128,256]):
        super().__init__()
        #self.SConv3d = SConv3d(1,n_fmaps[0],3,padding=1,bias=True)
        self.in_block = Conv(n_channels_img, n_fmaps[0],norm_type=norm_type,soft=False)
        self.down1 = Down(n_fmaps[0], n_fmaps[1],norm_type=norm_type,soft=False)

        self.in_block_mask = Conv(n_channels_mask, n_fmaps[0],norm_type=norm_type,soft=False)
        self.down1_mask = Down(n_fmaps[0], n_fmaps[1],norm_type=norm_type,soft=False)
        self.merge = Conv(n_fmaps[1], n_fmaps[1],norm_type=norm_type,soft=False)
        self.down2 = Down(n_fmaps[1], n_fmaps[2],norm_type=norm_type,soft=False)
        self.down3 = Down(n_fmaps[2], n_fmaps[3],norm_type=norm_type,soft=False)
        self.down4 = Down(n_fmaps[3], n_fmaps[4],norm_type=norm_type,soft=False)

        self.up2 = Up(n_fmaps[4],n_fmaps[3],norm_type=norm_type,soft=False)
        self.up3 = Up(n_fmaps[3],n_fmaps[2],norm_type=norm_type,soft=False)
        self.up4 = Up(n_fmaps[2],n_fmaps[1],norm_type=norm_type,soft=False)
        self.up5 = Up(n_fmaps[1],n_fmaps[0],norm_type=norm_type,soft=False)
        self.out_block = torch.nn.Conv3d(n_fmaps[0], n_class, 3, padding=1)
        self.final = nn.Softmax(dim=1)
        self.n_class = n_class
    def forward(self, data_dict,in_key_img,in_key_mask,out_key):
        x_img = data_dict[in_key_img]
        x_mask = data_dict[in_key_mask]
        #input_res = data_dict.get(self.in_key2)
        #input_x = self.SConv3d(input_x)
        x1_img = self.in_block(x_img)
        x1_mask = self.in_block_mask(x_mask)
        x2_img = self.down1(x1_img)
        x2_mask = self.down1_mask(x1_mask)
        x2 = x2_img+x2_mask
        x2 = self.merge(x2)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up2(x5)
        x = self.up3(x)+x3
        x = self.up4(x)+x2
        x = self.up5(x)
        x = self.out_block(x)
        x = self.final(x)
        data_dict[out_key] = x
        return data_dict

class Joint(torch.nn.Module):
    def __init__(self, models, vae_forward_scale=0.0, vae_decoder_dropout=0.0, seg_dropout=0.0):
        super().__init__()
        self.Seg = models[0]
        self.Vae = models[1]
        self.vae_forward_scale = vae_forward_scale
        self.vae_decoder_dropout = vae_decoder_dropout
        self.seg_dropout = seg_dropout

    def forward(self, data_dict,in_key,out_key,out_key_recon,dropout=False):
        if dropout: data_dict = self.Seg(data_dict,in_key,out_key, dropout=self.seg_dropout)
        else: data_dict = self.Seg(data_dict,in_key,out_key)
        if dropout: data_dict[out_key_recon],_,_ = self.Vae(data_dict[out_key],if_random=False,scale=self.vae_forward_scale,dropout=self.vae_decoder_dropout)
        else: data_dict[out_key_recon],data_dict["mean"],data_dict["std"] = self.Vae(data_dict[out_key],if_random=False,scale=self.vae_forward_scale)
        return data_dict


class Joint2(torch.nn.Module):
    def __init__(self, models, seg_dropout=0.0):
        super().__init__()
        self.Seg = models[0]
        self.Dis = models[1]
        self.seg_dropout = seg_dropout

    def forward(self, data_dict,in_key,out_key,score_key,dropout=False):
        if dropout: data_dict = self.Seg(data_dict,in_key,out_key, dropout=self.seg_dropout)
        else: data_dict = self.Seg(data_dict,in_key,out_key)
        data_dict[score_key] = self.Dis(data_dict[out_key][:,1:2,:,:,:])
        return data_dict


class Embed(torch.nn.Module):
    def __init__(self, models):
        super().__init__()
        self.Encoder = models[0]
        self.Vae = models[1]
        self.Fusion = models[2]
    def forward(self, data_dict,in_key,out_key,test_mode=False,loop_input=None,seg_input=None,latent_input=None):
        if latent_input:
            data_dict['latent_code'] = data_dict[latent_input]
        else:
            data_dict['latent_code'] = self.Encoder(data_dict[in_key])
        data_dict['gt_recon'],data_dict['latent_code_gt'],data_dict['latent_code_std'] = self.Vae(data_dict['venous_pancreas_only'],if_random=True,scale=0.5,mid_input=False)
        if loop_input:
            data_dict[loop_input],data_dict['latent_code_loop'],_ = self.Vae(data_dict[loop_input],if_random=False,scale=0,mid_input=False)
        if seg_input:
            data_dict['init_seg'] = data_dict[seg_input]
        else:
            data_dict['init_seg'] = self.Vae(data_dict['latent_code'],if_random=False,scale=0,mid_input=True)

        if loop_input:
            #T = 2*torch.sum(data_dict['venous_pancreas_only']*data_dict[loop_input],(2,3,4))/(torch.sum(data_dict[loop_input],(2,3,4))+torch.sum(data_dict['venous_pancreas_only'],(2,3,4)))
            #data_dict = self.Fusion(data_dict,in_key,loop_input,out_key,T[:,1].detach().view(-1,1,1,1,1))
            data_dict = self.Fusion(data_dict,in_key,loop_input,out_key)
        else:
            if test_mode:
                #T = 2*torch.sum(data_dict['venous_pancreas_only']*data_dict['init_seg'],(2,3,4))/(torch.sum(data_dict['init_seg'],(2,3,4))+torch.sum(data_dict['venous_pancreas_only'],(2,3,4)))
                data_dict = self.Fusion(data_dict,in_key,'init_seg',out_key)
            else:
                #T = 2*torch.sum(data_dict['venous_pancreas_only']*data_dict['gt_recon'],(2,3,4))/(torch.sum(data_dict['gt_recon'],(2,3,4))+torch.sum(data_dict['venous_pancreas_only'],(2,3,4)))
                data_dict = self.Fusion(data_dict,in_key,'gt_recon',out_key)
        data_dict['seg_recon'],_,_ = self.Vae(data_dict['init_seg'].detach(),if_random=False,scale=0,mid_input=False)
        #print(T[:,1].detach())
        return data_dict