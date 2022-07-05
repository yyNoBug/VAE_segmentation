import torch
import torch.nn as nn
import numpy as np
import pdb

def dice(A,B):
    return 2.0 * torch.sum(A*B)/(torch.sum(A)+torch.sum(B)+0.000001)

def binarize(A):
    return (A >= 0.5).float()

def confident_binarize(A, max=0.8, min=0.2):
    B = A.clone()
    B[B > max] = 1
    B[B < min] = 0
    # pdb.set_trace()
    # print(torch.abs(B - torch.where(A >= 0.5, 1, 0)).sum())
    return B


# def dic(A,B):
#     return torch.mean((2*torch.sum(A * B, (1,2,3)) / (torch.sum(A, (1,2,3))+torch.sum(B, (1,2,3))+0.000001)))


# a = torch.from_numpy(np.array([0.9,0.1]*8).reshape(2,2,2,2))
# b = torch.from_numpy(np.array([0.9,0.1]*8).reshape(2,2,2,2))
# print(dic(a,b))

def avg_ce(data_dict,source_key='align_lung', target_key='source_lung'):
    source_mask = data_dict[source_key]
    target_mask = data_dict[target_key]
    criterion = nn.BCELoss()

    if not isinstance(source_mask, list):
        source_mask = [source_mask]
    standard_loss_sum = 0
    for im in source_mask:
        standard_loss_sum += criterion(im,target_mask)
    return standard_loss_sum/len(source_mask)


def KLloss(data_dict,mean_key='mean',std_key='std'):
    Mean = data_dict[mean_key]
    Std = data_dict[std_key]
    return torch.mean(0.5*(torch.sum(torch.pow(Std,2),(1))+torch.sum(torch.pow(Mean,2),(1))-2*torch.sum(torch.log(Std+0.00001),(1))))


def avg_dsc(data_dict,source_key='align_lung', target_key='source_lung',binary=False,topindex=2,botindex=0,pad=[0,0,0],return_mean=True,detach=False):
    source_mask = data_dict[source_key]
    target_mask = data_dict[target_key]
    if not detach:
        target_mask = target_mask.cuda()
    else:
        target_mask = target_mask.cuda().detach()

    standard_loss_sum = 0

    if binary:
        label = (torch.argmax(source_mask,dim=1,keepdim=True)).type(torch.cuda.LongTensor)
        one_hot = torch.cuda.FloatTensor(label.size(0),source_mask.size(1),label.size(2),label.size(3),label.size(4)).zero_()
        source_mask = one_hot.scatter_(1,label.data,1)
        label = (torch.argmax(target_mask,dim=1,keepdim=True)).type(torch.cuda.LongTensor)
        one_hot = torch.cuda.FloatTensor(label.size(0),target_mask.size(1),label.size(2),label.size(3),label.size(4)).zero_()
        target_mask = one_hot.scatter_(1,label.data,1)
    else:
        source_mask = source_mask.cuda()

    if source_mask.shape[1]>1:
        #standard_loss_sum = standard_loss_sum + dice(source_mask[:,1:2,...],target_mask[:,1:2,...])
        #standard_loss_sum = standard_loss_sum + dice(source_mask[:,2:3,...],target_mask[:,2:3,...])
        if return_mean:
            standard_loss_sum +=  torch.mean((2*torch.sum(source_mask*target_mask,(2,3,4))/(torch.sum(source_mask,(2,3,4))+torch.sum(target_mask,(2,3,4))+0.000001))[:,botindex:topindex,...])
        else:
            standard_loss_sum +=  torch.mean((2*torch.sum(source_mask*target_mask,(2,3,4))/(torch.sum(source_mask,(2,3,4))+torch.sum(target_mask,(2,3,4))+0.000001))[:,botindex:topindex,...],1)
    else:
        if return_mean:
            standard_loss_sum += torch.mean(2*torch.sum(source_mask*target_mask,(2,3,4))/(torch.sum(source_mask,(2,3,4))+torch.sum(target_mask,(2,3,4))+0.000001))
        else:
            standard_loss_sum += torch.mean(2*torch.sum(source_mask*target_mask,(2,3,4))/(torch.sum(source_mask,(2,3,4))+torch.sum(target_mask,(2,3,4))+0.000001),1)
    return standard_loss_sum