import shutil
from utils.utils import MySpatialTransform
# from batchgenerators.transforms.spatial_transforms import SpatialTransform
import torch
import torch.nn as nn
import torchvision
import argparse
import random
from tensorboardX import SummaryWriter
import os
import importlib
import json
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import MSELoss,L1Loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from scipy.ndimage.morphology import binary_dilation, binary_erosion
from utils.utils import plot_slides,BaseDataset,NumpyLoader_Multi, NumpyLoader_Multi_merge,NiiLoader, image_resize,CropResize,CopyField, ExtendSqueeze,Reshape, PadToSize, Clip, Binarize, CenterIntensities
from utils.evaluation import binarize
import random
from utils.saver import Saver
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument("prefix", help="prefix")
parser.add_argument("-P","--target_phase", help="target_phase", default='arterial')
parser.add_argument("-G","--GPU", help="GPU", default='0,1,2,3')
parser.add_argument("-b","--batch_size", type=int, help="batch_size",default=4)
parser.add_argument("-E","--max_epoch", type=int, help="max_epoch",default=1600)
parser.add_argument("--save_epoch", type=int, help="save_epoch",default=50)
parser.add_argument("--eval_epoch", type=int, help="eval_epoch",default=50)
parser.add_argument("--turn_epoch", type=int, help="turn_epoch",default=-1)
parser.add_argument("-S","--softrelu", type=int, help="softrelu",default=0)
parser.add_argument("-M","--method", help="method", default='vae_train')
parser.add_argument("-R","--data_root", help="data_root", default='../nih_data/numpy_data/')
parser.add_argument("-V","--val_data_root", help="val_data_root", default='../nih_data/numpy_data/')
parser.add_argument("-l","--data_path", help="data_path", default='Multi_all.json')
parser.add_argument("-t","--train_list", help="train_list", default='NIH_train')
parser.add_argument("-v","--val_list", help="val_list", default='NIH_val')
parser.add_argument("--load_prefix", help="load_prefix", default=None)
parser.add_argument("--checkpoint_name", help="checkpoint_name", default="best_model.ckpt")
parser.add_argument("--load_prefix_vae", help="load_prefix_vae", default=None)
parser.add_argument("--load_prefix_joint", help="load_prefix_joint", default=None)
parser.add_argument("--pan_index", help="pan_index", default='1')
parser.add_argument("--lambda_vae", type=float, help="lambda_vae", default=0.1)
parser.add_argument("--lambda_vae_warmup", type=int, help="save_epoch",default=0)
parser.add_argument("--lr_seg", type=float, help="lr_seg", default=1e-2) # for seg 1e-1
parser.add_argument("--lr_vae", type=float, help="lr_vae", default=0) # for vae 1e-1
parser.add_argument("--test_only", help="test_only", action='store_true')
parser.add_argument("--resume", help="resume", action='store_true')
parser.add_argument("--save_more_reference", help="save_more_reference", action='store_true')
parser.add_argument("--save_eval_result", help="save_more_reference", action='store_true')
parser.add_argument("--no_aug", help="no_aug", action='store_true')
parser.add_argument("--adam", help="no_aug", action='store_true')
parser.add_argument("--mode", help="mode", type=int, default=0)
args = parser.parse_args()

data_root = args.data_root
val_data_root = args.val_data_root
lr1 = args.lr_seg # for seg 1e-1
lr2 = args.lr_vae # for vae 1e-1
train_list = args.train_list
softrelu = args.softrelu
val_list = args.val_list
torch.backends.cudnn.benchmark = True
weight_decay = 0
num_workers = 16
trainbatch = args.batch_size
valbatch = 1
load_prefix = args.load_prefix
checkpoint_name = args.checkpoint_name
load_prefix_vae = args.load_prefix_vae
load_prefix_joint = args.load_prefix_joint
load_epoch_seg = 240
load_epoch = 60
prefix = args.prefix
data_path = os.path.join('lists',args.data_path)
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
save_root_path = '3dmodel'
save_path = '3dmodel/'+prefix
display_path = 'tensorboard/'+prefix
middle_path = 'domain_cache/'+prefix
result_path = 'result/'+prefix
max_epoch = args.max_epoch
save_epoch = args.save_epoch
eval_epoch = args.eval_epoch
turn_epoch = args.turn_epoch
assert save_epoch % eval_epoch == 0
assert turn_epoch % eval_epoch == 0 or turn_epoch == -1
pan_index = args.pan_index
if pan_index != '10':
    mask_index = [[0,0]] + [[int(f), idx+1] for idx,f in enumerate(pan_index.split(','))]
else:
    mask_index = [[0,0], [[1,2], 1]]
target_phase = args.target_phase
lambda_vae = args.lambda_vae
lambda_vae_warmup = args.lambda_vae_warmup
test_only = args.test_only
resume = args.resume
method = args.method
save_more_reference = args.save_more_reference
save_eval_result = args.save_eval_result
if save_eval_result and not os.path.exists(result_path):
    os.mkdir(result_path)
no_aug = args.no_aug
adam = args.adam

mode = args.mode
input_phases=['venous']
output_keys=['venous']


input_phases_mask =  input_phases + [f+'_mask' for f in input_phases] + [f+'_origin' for f in input_phases]+ [f+'_lung' for f in input_phases] + [f+'_pancreas' for f in input_phases] 

input_size=[256,256,256]
patch_size=[128,128,128]

pad_size = [0,0,0]
## define trainer myself


def filedict_from_json(json_path, key, epoch=1):
    with open(json_path, 'r') as f:
        json_dict = json.load(f)

    listdict = json_dict.get(key, [])
    output = []
    for i in range(epoch):
        output += listdict
    return output

def dice(A,B):
    return 2.0 * torch.sum(A*B)/(torch.sum(A)+torch.sum(B)+0.000001)
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
            standard_loss_sum +=  torch.mean((2*torch.sum(source_mask*target_mask,(2,3,4))/(torch.sum(source_mask,(2,3,4))+torch.sum(target_mask,(2,3,4))+0.0001))[:,botindex:topindex,...])
        else:
            standard_loss_sum +=  torch.mean((2*torch.sum(source_mask*target_mask,(2,3,4))/(torch.sum(source_mask,(2,3,4))+torch.sum(target_mask,(2,3,4))+0.0001))[:,botindex:topindex,...],1)
    else:
        if return_mean:
            standard_loss_sum += torch.mean(2*torch.sum(source_mask*target_mask,(2,3,4))/(torch.sum(source_mask,(2,3,4))+torch.sum(target_mask,(2,3,4))+0.0001))
        else:
            standard_loss_sum += torch.mean(2*torch.sum(source_mask*target_mask,(2,3,4))/(torch.sum(source_mask,(2,3,4))+torch.sum(target_mask,(2,3,4))+0.0001),1)
    return standard_loss_sum

if __name__ == "__main__":
    ## dataset
    train_data_list = filedict_from_json(data_path, train_list, eval_epoch)
    # print(train_data_list)

    transforms = {'train': []}
    ## define training data pipeline
    transforms['train'].append(NumpyLoader_Multi_merge(fields=input_phases, root_dir=data_root, load_mask=True, mask_index=mask_index))
    transforms['train'].append(CropResize(fields=input_phases, output_size=patch_size))
    #transforms['train'].append(PadToSize(fields=input_phases, size=input_size, pad_val=-1024, seg_pad_val=0,random_subpadding=True,load_mask=method=='seg_train'))
    
    if not no_aug:
        transforms['train'].append(Reshape(fields=input_phases_mask))
        transforms['train'].append(MySpatialTransform(patch_size,[dis//2-5 for dis in patch_size], random_crop=True,
                    scale=(0.85,1.15),
                    do_elastic_deform=False, alpha=(0,500),
                    do_rotation=True, sigma=(10,30.),
                    angle_x=(-0.2,0.2), angle_y=(-0.2, 0.2),
                    angle_z=(-0.2, 0.2),
                    border_mode_data="constant",
                    border_cval_data=-1024,
                    data_key="venous", p_el_per_sample=0,label_key="venous_pancreas",
                    p_scale_per_sample=1, p_rot_per_sample=1))
    
    #transforms['train'].append(PadToSize(fields=input_phases, size=[1,1]+patch_size, pad_val=-1024, seg_pad_val=0,random_subpadding=True,load_mask=method=='seg_train'))
    for phase in input_phases:
        transforms['train'].append(CopyField(fields=[phase], to_field=[phase+'_origin']))
    transforms['train'].append(Clip(fields=input_phases,new_min=-200, new_max=400))
    transforms['train'].append(CenterIntensities(fields=input_phases,subtrahend=100, divisor=300))
    transforms['train'].append(Reshape(fields=input_phases_mask, reshape_view=[-1]+patch_size))

    val_data_list = filedict_from_json(data_path, val_list)
    transforms['val']=[]
    ## define validation data pipeline
    transforms['val'].append(NumpyLoader_Multi_merge(fields=input_phases, root_dir=val_data_root,load_mask=True,mask_index=mask_index))
    transforms['val'].append(CropResize(fields=input_phases,output_size=patch_size))
    #transforms['val'].append(PadToSize(fields=input_phases, size=input_size, pad_val=-1024, seg_pad_val=0,random_subpadding=False,load_mask=True))
    for phase in input_phases:
        transforms['val'].append(CopyField(fields=[phase], to_field=[phase+'_origin']))
    transforms['val'].append(Clip(fields=input_phases,new_min=-200, new_max=400))
    transforms['val'].append(CenterIntensities(fields=input_phases,subtrahend=100, divisor=300))
    transforms['val'].append(Reshape(fields=input_phases_mask, reshape_view=[-1]+patch_size))

    for k,v in transforms.items():
        transforms[k] = torchvision.transforms.Compose(v)

    ###############################################################################################
    ############################ Create Datasets ##################################################
    ###############################################################################################
    print("Loading data.")
    train_dataset = BaseDataset(train_data_list, transforms=transforms['train'])
    val_dataset = BaseDataset(val_data_list, transforms=transforms['val'])
    if method != "domain_adaptation":
        train_loader = DataLoader(train_dataset, batch_size=trainbatch, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=trainbatch, shuffle=False, num_workers=num_workers, drop_last=True, pin_memory=True)
        print("domain!")
    val_loader = DataLoader(val_dataset, batch_size=valbatch, shuffle=False, num_workers=num_workers, pin_memory=True)
    if save_more_reference:
        train_loader_2 = DataLoader(train_dataset, batch_size=valbatch, shuffle=False, num_workers=num_workers, pin_memory=True)

    ## model build and load
    print("Building model.")
    models = importlib.import_module('joint_model')
    # vm_model = importlib.import_module('models.' + 'voxelmorph3D_joint')
    if method=='vae_train':
        model=models.VAE(n_channels=len(mask_index), n_class=len(mask_index),norm_type=1,dim=128,soft=softrelu==1)
    elif method=='seg_train':
        model=models.Segmentation(n_channels=1, n_class=len(mask_index),norm_type=1)
        model_ref = models.VAE(n_channels=len(mask_index), n_class=len(mask_index),norm_type=1,dim=128)
    elif method=='joint_train' or method == "domain_adaptation":
        model=[]
        model.append(models.Segmentation(n_channels=1, n_class=len(mask_index),norm_type=1))
        model.append(models.VAE(n_channels=len(mask_index), n_class=len(mask_index),norm_type=1,dim=128))
        model = models.Joint(models=model)
    elif method=='embed_train' or method=='refine_vae':
        model=[]
        model.append(models.Encoder(n_channels=1, dim=128, norm_type=1))
        model.append(models.VAE(n_channels=len(mask_index), n_class=len(mask_index),norm_type=1,dim=128))
        model.append(models.Fusion(n_channels_img=1,n_channels_mask=len(mask_index), n_class=len(mask_index),norm_type=1))
        model = models.Embed(models=model)
    elif method=='sep_joint_train':
        model=[]
        model.append(models.Segmentation(n_channels=1, n_class=len(mask_index),norm_type=1))
        model.append(models.VAE(n_channels=len(mask_index), n_class=len(mask_index),norm_type=1,dim=128))
        model = models.Joint(models=model)
        tea_model=[]
        tea_model.append(models.Segmentation(n_channels=1, n_class=len(mask_index),norm_type=1))
        tea_model.append(models.VAE(n_channels=len(mask_index), n_class=len(mask_index),norm_type=1,dim=128))
        tea_model = models.Joint(models=tea_model)
    else:
        raise ValueError("Try a valid method.")
    model = model.cuda()

    if method!='joint_train':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=lr1,weight_decay = weight_decay,momentum=0.9)
    else:
        '''
        optimizer = torch.optim.SGD([{'params':model.Seg.down3.parameters(),'lr':lr1,'model':'Seg'},
                                    {'params':model.Seg.down4.parameters(),'lr':lr1,'model':'Seg'},
                                        {'params':model.Vae.parameters(),'lr':lr2,'model':'Vae'} ],
                                    weight_decay = weight_decay)
        
        '''
        if not adam: optimizer = torch.optim.SGD([{'params':model.Seg.parameters(),'lr':lr1,'model':'Seg'},
                                        {'params':model.Vae.parameters(),'lr':lr2,'model':'Vae'} ],
                            weight_decay = weight_decay,momentum=0.9)
        else: optimizer = torch.optim.Adam([{'params':model.Seg.parameters(),'lr':lr1,'model':'Seg'},
                                        {'params':model.Vae.parameters(),'lr':lr2,'model':'Vae'} ],
                            betas=(0.9, 0.999),weight_decay = weight_decay)

        

    
    print("Loading prefix.")
    if load_prefix:
        register_model_path = os.path.join(save_root_path, load_prefix, checkpoint_name)
        if method=="seg_train":
            model.load_state_dict(torch.load(register_model_path)['model_state_dict'])
        else:
            model.Seg.load_state_dict(torch.load(register_model_path)['model_state_dict'])

    if load_prefix_vae:
        register_model_path = save_root_path+'/'+load_prefix_vae+'/best_model.ckpt'
        if method=="seg_train":
            model_ref.load_state_dict(torch.load(register_model_path)['model_state_dict'])
            ref_model_parallel = nn.DataParallel(model_ref).cuda()
            for param in ref_model_parallel.parameters():
                param.requires_grad = False
            model_ref.eval()
        else:
            model.Vae.load_state_dict(torch.load(register_model_path)['model_state_dict'])
        '''
        pretrained_dict = torch.load(register_model_path)['model_state_dict']
        model_dict = register_model.segmentation_model.state_dict()
        pretrained_dict = {k.split('.',1)[1]: v for k, v in pretrained_dict.items() if k.split('.',1)[0]=='segmentation_model' and k.split('.',1)[1] in model_dict}
        model_dict.update(pretrained_dict) 
        register_model.segmentation_model.load_state_dict(model_dict)
        register_model.segmentation_model.eval()
        '''
    
    if load_prefix_joint:
        register_model_path = save_root_path+'/'+load_prefix_joint+'/best_model.ckpt'
        model.load_state_dict(torch.load(register_model_path)['model_state_dict'])
    
    if method=='sep_joint_train':
        if load_prefix_joint:
            register_model_path = save_root_path+'/'+load_prefix_joint+'/best_model.ckpt'
            tea_model.load_state_dict(torch.load(register_model_path)['model_state_dict'])
        else:
            register_model_path = save_root_path+'/'+load_prefix+'/best_model.ckpt'
            tea_model.Seg.load_state_dict(torch.load(register_model_path)['model_state_dict'])
            register_model_path = save_root_path+'/'+load_prefix_vae+'/best_model.ckpt'
            tea_model.Vae.load_state_dict(torch.load(register_model_path)['model_state_dict'])
        tea_model_parallel = nn.DataParallel(tea_model).cuda()
        for param in tea_model.parameters():
            param.requires_grad = False
    
    if method=='joint_train' or method=='sep_joint_train' or method=='embed_train' or method=='domain_adaptation':
        for param in model.Vae.parameters():
            param.requires_grad = False
        model.Vae.eval()
    if method=='refine_vae':
        Encoder_list = ['in_block','down1','down2','down3','down4','down5','fc_mean','fc_std']
        for param_name,param in model.Vae.named_parameters():
            if param_name.split('.')[0] in Encoder_list:
                param.requires_grad = False
            else:
                param.requires_grad = True
    final_model_parallel = nn.DataParallel(model).cuda()
    label_key = 'venous_pancreas'
    img_key = 'venous'
    best_result = 0
    train_dis = 0
    max_idx_in_epoch = 0
    saver = Saver(display_dir=display_path,display_freq=10)
    MSE_Loss = MSELoss()
    
    ## training loop 
    print("Start training")
    for epoch in range(max_epoch // eval_epoch):
        if not test_only:
            if epoch == 0 and method == "domain_adaptation":
                if not os.path.exists(middle_path): 
                    os.mkdir(middle_path)
                for idx, batch in enumerate(train_loader):
                    batch[label_key+'_only'] = batch[label_key].type(torch.cuda.LongTensor)
                    one_hot = torch.cuda.FloatTensor(batch[label_key+'_only'].size(0),len(mask_index),batch[label_key+'_only'].size(2),batch[label_key+'_only'].size(3),batch[label_key+'_only'].size(4)).zero_()
                    batch[label_key+'_only'] = one_hot.scatter_(1,batch[label_key+'_only'].data,1)
                    batch[img_key] = batch[img_key].cuda()
                    batch = final_model_parallel(batch, img_key, label_key+'_pred', label_key+'_recon_pred')
                    filename = os.path.join(middle_path, f'{idx}_pred.pt')
                    torch.save(batch[label_key+'_pred'], filename)
                    filename = os.path.join(middle_path, f'{idx}_recon.pt')
                    torch.save(batch[label_key+'_recon_pred'], filename)
                    
            for idx, batch in enumerate(train_loader):
                if idx > max_idx_in_epoch:
                    max_idx_in_epoch = idx
                #optimizer.param_groups[0]['lr'] = lr3/(10**(epoch//10))
                #for out_list in range(len(output_keys)):

                optimizer.zero_grad()
                # forward + backward + optimize
                if method =='vae_train':
                    batch[label_key+'_only'] = batch[label_key].type(torch.cuda.LongTensor)
                    one_hot = torch.cuda.FloatTensor(batch[label_key+'_only'].size(0),len(mask_index),batch[label_key+'_only'].size(2),batch[label_key+'_only'].size(3),batch[label_key+'_only'].size(4)).zero_()
                    batch[label_key+'_only'] = one_hot.scatter_(1,batch[label_key+'_only'].data,1)
                    batch[label_key+'_recon'],batch['mean'],batch['std']  = final_model_parallel(batch[label_key+'_only'],if_random=True,scale=0.35) #0.2
                    h=batch[label_key+'_only'][0:1,0:1,:,:,:].shape[4]
                    batch[label_key+'_display'] = torch.cat((batch[label_key+'_only'][0:1,0:1,:,:,h//2], \
                            batch[label_key+'_only'][0:1,1:2,:,:,h//2],batch[label_key+'_recon'][0:1,1:2,:,:,h//2]), dim=0)
                        
                    #batch = register_model(batch)
                    '''
                    final_loss = standard_loss(batch, do_mask=True, source_label_key='align_img', target_label_key='dummy_img', mask_label_key='source_mask')*(1-beta) + \
                                (standard_loss(batch, do_mask=True, source_label_key='target', target_label_key='dummy_img', mask_label_key='source_mask') + \
                                standard_loss(batch, do_mask=True, source_label_key='align_img', target_label_key='source', mask_label_key='source_mask'))*beta/2
                    '''
                    klloss = KLloss(batch)
                    dsc_loss = 1-avg_dsc(batch,source_key=label_key+'_recon', target_key=label_key+'_only',botindex=1,topindex=len(mask_index))
                    final_loss = dsc_loss+0.00002*klloss
                    loss = []
                    display_image={}
                    loss.append(['dice_loss',dsc_loss.item()])
                    loss.append(['kl_loss',klloss.item()])
                    loss.append(['lr',optimizer.param_groups[0]['lr']])
                    display_image.update({label_key+'_display':batch[label_key+'_display']})
                    saver.write_display(idx+epoch*(max_idx_in_epoch+1),loss,display_image)

                if method =='seg_train':
                    if epoch == 0: continue
                    batch[label_key+'_only'] = batch[label_key].type(torch.cuda.LongTensor)
                    one_hot = torch.cuda.FloatTensor(batch[label_key+'_only'].size(0),len(mask_index),batch[label_key+'_only'].size(2),batch[label_key+'_only'].size(3),batch[label_key+'_only'].size(4)).zero_()
                    batch[label_key+'_only'] = one_hot.scatter_(1,batch[label_key+'_only'].data,1)
                    batch[img_key] = batch[img_key].cuda()
                    batch = final_model_parallel(batch, img_key, label_key+'_pred')
                    if load_prefix_vae is not None:
                        batch[label_key+'_recon_pred'],_,_ = ref_model_parallel(batch[label_key+'_pred'],if_random=False,scale=0)

                    h=batch[label_key+'_pred'][0:1,0:1,:,:,:].shape[4]
                    batch[label_key+'_display']= torch.cat((batch[img_key][0:1,0:1,:,:,h//2], \
                            batch[label_key+'_only'][0:1,1:2,:,:,h//2],batch[label_key+'_pred'][0:1,1:2,:,:,h//2]), dim=0)
                        
                    #batch = register_model(batch)
                    '''
                    final_loss = standard_loss(batch, do_mask=True, source_label_key='align_img', target_label_key='dummy_img', mask_label_key='source_mask')*(1-beta) + \
                                (standard_loss(batch, do_mask=True, source_label_key='target', target_label_key='dummy_img', mask_label_key='source_mask') + \
                                standard_loss(batch, do_mask=True, source_label_key='align_img', target_label_key='source', mask_label_key='source_mask'))*beta/2
                    '''
                    
                    dsc_loss = 1 - avg_dsc(batch,source_key=label_key+'_pred', target_key=label_key+'_only',botindex=1,topindex=len(mask_index))
                    final_loss = dsc_loss
                    loss = []
                    display_image={}
                    loss.append(['dice_loss',dsc_loss.item()])
                    loss.append(['lr',optimizer.param_groups[0]['lr']])
                    if load_prefix_vae is not None:
                        recon_loss = 1 - avg_dsc(batch,source_key=label_key+'_pred', target_key=label_key+'_recon_pred',botindex=1,topindex=len(mask_index),return_mean=True)
                        loss.append(['recon_loss',recon_loss.item()])
                    display_image.update({label_key+'_display':batch[label_key+'_display']})
                    saver.write_display(idx+epoch*(max_idx_in_epoch+1),loss,display_image)

                if method =='joint_train':
                    batch[label_key+'_only'] = batch[label_key].type(torch.cuda.LongTensor)
                    one_hot = torch.cuda.FloatTensor(batch[label_key+'_only'].size(0),len(mask_index),batch[label_key+'_only'].size(2),batch[label_key+'_only'].size(3),batch[label_key+'_only'].size(4)).zero_()
                    batch[label_key+'_only'] = one_hot.scatter_(1,batch[label_key+'_only'].data,1)
                    batch[img_key] = batch[img_key].cuda()
                    batch = final_model_parallel(batch,img_key,label_key+'_pred',label_key+'_recon_pred')
                    h=batch[label_key+'_pred'][0:1,0:1,:,:,:].shape[4]
                    batch[label_key+'_display']= torch.cat((batch[label_key+'_recon_pred'][0:1,1:2,:,:,h//2], \
                            batch[label_key+'_only'][0:1,1:2,:,:,h//2],batch[label_key+'_pred'][0:1,1:2,:,:,h//2]), dim=0)
                        
                    #batch = register_model(batch)
                    '''
                    final_loss = standard_loss(batch, do_mask=True, source_label_key='align_img', target_label_key='dummy_img', mask_label_key='source_mask')*(1-beta) + \
                                (standard_loss(batch, do_mask=True, source_label_key='target', target_label_key='dummy_img', mask_label_key='source_mask') + \
                                standard_loss(batch, do_mask=True, source_label_key='align_img', target_label_key='source', mask_label_key='source_mask'))*beta/2
                    '''
                    '''
                    recon_loss = avg_dsc(batch,source_key=label_key+'_pred', target_key=label_key+'_recon_pred',botindex=1,topindex=len(mask_index),return_mean=False)
                    dsc_loss = avg_dsc(batch,source_key=label_key+'_pred', target_key=label_key+'_pred',botindex=1,topindex=len(mask_index),return_mean=False,detach=True)
                    final_loss = 1-torch.mean(recon_loss) + 1 - torch.mean(dsc_loss*(recon_loss.detach()))
                    '''
                    recon_loss = 1 - avg_dsc(batch,source_key=label_key+'_pred', target_key=label_key+'_recon_pred',botindex=1,topindex=len(mask_index),return_mean=True)
                    dsc_loss = 1 - avg_dsc(batch,source_key=label_key+'_pred', target_key=label_key+'_only',botindex=1,topindex=len(mask_index),return_mean=True)
                    final_loss = lambda_vae * recon_loss + dsc_loss
                    loss = []
                    display_image={}
                    loss.append(['recon_loss',recon_loss.item()])
                    loss.append(['dice_loss',dsc_loss.item()])
                    loss.append(['lr',optimizer.param_groups[0]['lr']])
                    display_image.update({label_key+'_display':batch[label_key+'_display']})
                    saver.write_display(idx+epoch*(max_idx_in_epoch+1),loss,display_image)

                if method =='domain_adaptation':
                    if epoch == 0: continue
                    batch[label_key+'_only'] = batch[label_key].type(torch.cuda.LongTensor)
                    one_hot = torch.cuda.FloatTensor(batch[label_key+'_only'].size(0),len(mask_index),batch[label_key+'_only'].size(2),batch[label_key+'_only'].size(3),batch[label_key+'_only'].size(4)).zero_()
                    batch[label_key+'_only'] = one_hot.scatter_(1,batch[label_key+'_only'].data,1)
                    batch[img_key] = batch[img_key].cuda()
                    batch = final_model_parallel(batch,img_key,label_key+'_pred',label_key+'_recon_pred')
                    h=batch[label_key+'_pred'][0:1,0:1,:,:,:].shape[4]

                    filename = os.path.join(middle_path, f'{idx}_pred.pt')
                    dat = torch.load(filename)
                    batch[label_key+'_only_fake'] = dat
                    # one_hot = torch.cuda.FloatTensor(batch[label_key+'_only'].size(0),len(mask_index),batch[label_key+'_only'].size(2),batch[label_key+'_only'].size(3),batch[label_key+'_only'].size(4)).zero_()
                    # batch[label_key+'_only_fake'] = one_hot.scatter_(1,batch[label_key+'_only_fake'].data,1)
                    batch[label_key+'_display']= torch.cat((batch[label_key+'_recon_pred'][0:1,1:2,:,:,h//2], \
                            batch[label_key+'_only'][0:1,1:2,:,:,h//2],batch[label_key+'_pred'][0:1,1:2,:,:,h//2], \
                            batch[label_key+'_only_fake'][0:1,1:2,:,:,h//2]), dim=0)
                    
                    # filename = os.path.join(middle_path, f'{idx}_try.pt')
                    # torch.save(batch[label_key+'_pred'], filename)
                    # tt = torch.load(filename)
                    # # with open(filename, 'wb') as f:
                    # #     np.save(f, batch[label_key+'_pred'].cpu().detach().numpy())
                    # # with open(filename, 'rb') as f:
                    # #     tt = np.load(f)
                    # batch[label_key+'_try'] = tt
                    # print(avg_dsc(batch, source_key=label_key+'_pred', target_key=label_key+'_try',botindex=1,topindex=len(mask_index),return_mean=True))

                    if mode != 0 and epoch % mode == 0:
                        filename = os.path.join(middle_path, f'{idx}_pred.pt')
                        torch.save(batch[label_key+'_pred'], filename)
                    
                    #batch = register_model(batch)
                    '''
                    final_loss = standard_loss(batch, do_mask=True, source_label_key='align_img', target_label_key='dummy_img', mask_label_key='source_mask')*(1-beta) + \
                                (standard_loss(batch, do_mask=True, source_label_key='target', target_label_key='dummy_img', mask_label_key='source_mask') + \
                                standard_loss(batch, do_mask=True, source_label_key='align_img', target_label_key='source', mask_label_key='source_mask'))*beta/2
                    '''
                    '''
                    recon_loss = avg_dsc(batch,source_key=label_key+'_pred', target_key=label_key+'_recon_pred',botindex=1,topindex=len(mask_index),return_mean=False)
                    dsc_loss = avg_dsc(batch,source_key=label_key+'_pred', target_key=label_key+'_pred',botindex=1,topindex=len(mask_index),return_mean=False,detach=True)
                    final_loss = 1-torch.mean(recon_loss) + 1 - torch.mean(dsc_loss*(recon_loss.detach()))
                    '''
                    recon_loss = 1 - avg_dsc(batch,source_key=label_key+'_pred', target_key=label_key+'_recon_pred',botindex=1,topindex=len(mask_index),return_mean=True)
                    dsc_loss = 1 - avg_dsc(batch,source_key=label_key+'_pred', target_key=label_key+'_only',botindex=1,topindex=len(mask_index),return_mean=True)
                    dsc_loss_fake = 1 - avg_dsc(batch,source_key=label_key+'_pred', target_key=label_key+'_only_fake',botindex=1,topindex=len(mask_index),return_mean=True)

                    if turn_epoch != -1:
                        if (epoch // turn_epoch) % 2 == 0:
                            final_loss = 2 * lambda_vae * recon_loss
                        else:
                            final_loss = lambda_vae * recon_loss + dsc_loss_fake
                    elif epoch >= lambda_vae_warmup:
                        final_loss = lambda_vae * recon_loss + dsc_loss_fake
                    else:
                        final_loss = lambda_vae * epoch / lambda_vae_warmup * recon_loss + dsc_loss_fake
                    loss = []
                    display_image={}
                    loss.append(['recon_loss',recon_loss.item()])
                    loss.append(['dice_loss_fake',dsc_loss_fake.item()])
                    loss.append(['dice_loss',dsc_loss.item()])
                    loss.append(['final_loss',final_loss.item()])
                    loss.append(['lr',optimizer.param_groups[0]['lr']])
                    display_image.update({label_key+'_display':batch[label_key+'_display']})
                    saver.write_display(idx+epoch*(max_idx_in_epoch+1),loss,display_image)

                if method =='embed_train':
                    batch[label_key+'_only'] = batch[label_key].type(torch.cuda.LongTensor)
                    one_hot = torch.cuda.FloatTensor(batch[label_key+'_only'].size(0),len(mask_index),batch[label_key+'_only'].size(2),batch[label_key+'_only'].size(3),batch[label_key+'_only'].size(4)).zero_()
                    batch[label_key+'_only'] = one_hot.scatter_(1,batch[label_key+'_only'].data,1)
                    batch[img_key] = batch[img_key].cuda()
                    for param in model.Encoder.parameters():
                        if epoch % 2 ==0:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                    batch = final_model_parallel(batch,img_key,label_key+'_pred',test_mode=True)

                    h=batch[label_key+'_pred'][0:1,0:1,:,:,:].shape[4]
                    batch[label_key+'_display']= torch.cat((batch[label_key+'_only'][0:1,1:2,:,:,h//2],batch[label_key+'_pred'][0:1,1:2,:,:,h//2],batch['init_seg'][0:1,1:2,:,:,h//2]), dim=0)
                        
                    #batch = register_model(batch)
                    '''
                    final_loss = standard_loss(batch, do_mask=True, source_label_key='align_img', target_label_key='dummy_img', mask_label_key='source_mask')*(1-beta) + \
                                (standard_loss(batch, do_mask=True, source_label_key='target', target_label_key='dummy_img', mask_label_key='source_mask') + \
                                standard_loss(batch, do_mask=True, source_label_key='align_img', target_label_key='source', mask_label_key='source_mask'))*beta/2
                    '''
                    '''
                    recon_loss = avg_dsc(batch,source_key=label_key+'_pred', target_key=label_key+'_recon_pred',botindex=1,topindex=len(mask_index),return_mean=False)
                    dsc_loss = avg_dsc(batch,source_key=label_key+'_pred', target_key=label_key+'_pred',botindex=1,topindex=len(mask_index),return_mean=False,detach=True)
                    final_loss = 1-torch.mean(recon_loss) + 1 - torch.mean(dsc_loss*(recon_loss.detach()))
                    '''

                    dsc_loss1 = 1-avg_dsc(batch,source_key=label_key+'_pred', target_key=label_key+'_only',botindex=1,topindex=len(mask_index))
                    dsc_loss2 = 1-avg_dsc(batch,source_key='init_seg', target_key=label_key+'_only',botindex=1,topindex=len(mask_index))

                    klloss = KLloss(batch,mean_key='latent_code_gt',std_key='latent_code_std')
                    recon_loss = 1-avg_dsc(batch,source_key='gt_recon', target_key=label_key+'_only',botindex=1,topindex=len(mask_index))
                    inpaint_loss = 1-avg_dsc(batch,source_key='seg_recon', target_key=label_key+'_only',botindex=1,topindex=len(mask_index))

                    mse_loss = MSE_Loss(batch['latent_code'],batch['latent_code_gt'])
                    final_loss = (dsc_loss1+dsc_loss2+inpaint_loss)/3 + mse_loss/10 + 0.00002*klloss + recon_loss
                    loss = []
                    display_image={}
                    loss.append(['dice_loss1',dsc_loss1.item()])
                    loss.append(['dice_loss2',dsc_loss2.item()])
                    loss.append(['mse_loss',mse_loss.item()])
                    loss.append(['lr',optimizer.param_groups[0]['lr']])
                    display_image.update({label_key+'_display':batch[label_key+'_display']})
                    saver.write_display(idx+epoch*(max_idx_in_epoch+1),loss,display_image)

                if method =='refine_vae':
                    batch[label_key+'_only'] = batch[label_key].type(torch.cuda.LongTensor)
                    one_hot = torch.cuda.FloatTensor(batch[label_key+'_only'].size(0),len(mask_index),batch[label_key+'_only'].size(2),batch[label_key+'_only'].size(3),batch[label_key+'_only'].size(4)).zero_()
                    batch[label_key+'_only'] = one_hot.scatter_(1,batch[label_key+'_only'].data,1)
                    batch[img_key] = batch[img_key].cuda()
                    for param in model.Encoder.parameters():
                        param.requires_grad = False
                    batch = final_model_parallel(batch,img_key,label_key+'_pred',test_mode=True)

                    h=batch[label_key+'_pred'][0:1,0:1,:,:,:].shape[4]
                    batch[label_key+'_display']= torch.cat((batch[label_key+'_only'][0:1,1:2,:,:,h//2],batch[label_key+'_pred'][0:1,1:2,:,:,h//2],batch['init_seg'][0:1,1:2,:,:,h//2]), dim=0)
                        
                    #batch = register_model(batch)
                    '''
                    final_loss = standard_loss(batch, do_mask=True, source_label_key='align_img', target_label_key='dummy_img', mask_label_key='source_mask')*(1-beta) + \
                                (standard_loss(batch, do_mask=True, source_label_key='target', target_label_key='dummy_img', mask_label_key='source_mask') + \
                                standard_loss(batch, do_mask=True, source_label_key='align_img', target_label_key='source', mask_label_key='source_mask'))*beta/2
                    '''
                    '''
                    recon_loss = avg_dsc(batch,source_key=label_key+'_pred', target_key=label_key+'_recon_pred',botindex=1,topindex=len(mask_index),return_mean=False)
                    dsc_loss = avg_dsc(batch,source_key=label_key+'_pred', target_key=label_key+'_pred',botindex=1,topindex=len(mask_index),return_mean=False,detach=True)
                    final_loss = 1-torch.mean(recon_loss) + 1 - torch.mean(dsc_loss*(recon_loss.detach()))
                    '''

                    klloss = KLloss(batch,mean_key='latent_code_gt',std_key='latent_code_std')
                    recon_loss = 1-avg_dsc(batch,source_key='gt_recon', target_key=label_key+'_only',botindex=1,topindex=len(mask_index))
                    inpaint_loss = 1-avg_dsc(batch,source_key='seg_recon', target_key=label_key+'_only',botindex=1,topindex=len(mask_index))
                    init_loss = 1-avg_dsc(batch,source_key='init_seg', target_key=label_key+'_only',botindex=1,topindex=len(mask_index))
                    final_loss = inpaint_loss + 0.00002*klloss + recon_loss
                    loss = []
                    display_image={}
                    loss.append(['recon_loss',recon_loss.item()])
                    loss.append(['inpaint_loss',inpaint_loss.item()])
                    loss.append(['kl_loss',klloss.item()])
                    loss.append(['lr',optimizer.param_groups[0]['lr']])
                    display_image.update({label_key+'_display':batch[label_key+'_display']})
                    saver.write_display(idx+epoch*(max_idx_in_epoch+1),loss,display_image)

                if method =='sep_joint_train':
                    batch[label_key+'_only'] = batch[label_key].type(torch.cuda.LongTensor)
                    one_hot = torch.cuda.FloatTensor(batch[label_key+'_only'].size(0),len(mask_index),batch[label_key+'_only'].size(2),batch[label_key+'_only'].size(3),batch[label_key+'_only'].size(4)).zero_()
                    batch[label_key+'_only'] = one_hot.scatter_(1,batch[label_key+'_only'].data,1)
                    batch[img_key] = batch[img_key].cuda()
                    batch = final_model_parallel(batch,img_key,label_key+'_pred',label_key+'_recon_pred')
                    batch = tea_model_parallel(batch,img_key,label_key+'_pred_tea',label_key+'_recon_pred_tea')

                    h=batch[label_key+'_pred'][0:1,0:1,:,:,:].shape[4]
                    batch[label_key+'_display']= torch.cat((batch[label_key+'_recon_pred'][0:1,1:2,:,:,h//2], \
                            batch[label_key+'_only'][0:1,1:2,:,:,h//2],batch[label_key+'_pred'][0:1,1:2,:,:,h//2]), dim=0)
                        
                    #batch = register_model(batch)
                    '''
                    final_loss = standard_loss(batch, do_mask=True, source_label_key='align_img', target_label_key='dummy_img', mask_label_key='source_mask')*(1-beta) + \
                                (standard_loss(batch, do_mask=True, source_label_key='target', target_label_key='dummy_img', mask_label_key='source_mask') + \
                                standard_loss(batch, do_mask=True, source_label_key='align_img', target_label_key='source', mask_label_key='source_mask'))*beta/2
                    '''
                    
                    recon_loss = avg_dsc(batch,source_key=label_key+'_pred', target_key=label_key+'_recon_pred',botindex=1,topindex=len(mask_index),return_mean=False)
                    recon_loss_tea = avg_dsc(batch,source_key=label_key+'_pred_tea', target_key=label_key+'_recon_pred_tea',botindex=1,topindex=len(mask_index),return_mean=False)
                    dsc_loss = avg_dsc(batch,source_key=label_key+'_pred', target_key=label_key+'_pred_tea',botindex=1,topindex=len(mask_index),return_mean=False)
                    final_loss = 0.1*(1-torch.mean(recon_loss)) + 1 - torch.mean(dsc_loss*((recon_loss_tea)**2))
                    loss = []
                    display_image={}
                    loss.append(['recon_loss',(1-torch.mean(recon_loss)).item()])
                    loss.append(['dice_loss',(1-torch.mean(dsc_loss)).item()])
                    loss.append(['lr',optimizer.param_groups[0]['lr']])
                    display_image.update({label_key+'_display':batch[label_key+'_display']})
                    saver.write_display(idx+epoch*(max_idx_in_epoch+1),loss,display_image)

                final_loss.backward()
                optimizer.step()
                # print statistics
                if method =='vae_train':
                    print('[%3d, %3d] loss: %.4f, %.4f' %
                            ((epoch+1)*eval_epoch, idx + 1, dsc_loss.item(),klloss.item()))
                if method =='seg_train':    
                    print('[%3d, %3d] loss: %.4f' %
                            ((epoch+1)*eval_epoch, idx + 1, dsc_loss.item()))
                if method =='joint_train':    
                    print('[%3d, %3d] loss: %.4f, %.4f' %
                            ((epoch+1)*eval_epoch, idx + 1, recon_loss.item(), dsc_loss.item()))
                if method == 'domain_adaptation':
                    print('[%3d, %3d] loss: %.4f, %.4f, %.4f' %
                            ((epoch+1)*eval_epoch, idx + 1, recon_loss.item(), dsc_loss_fake.item(), dsc_loss.item()))
                if method =='embed_train':    
                    print('[%3d, %3d] loss: %.4f, %.4f, %.4f, %.4f, %.4f' %
                            ((epoch+1)*eval_epoch, idx + 1, dsc_loss1.item(),dsc_loss2.item(),mse_loss.item(),inpaint_loss.item(),recon_loss.item()))
                if method =='refine_vae':    
                    print('[%3d, %3d] loss: %.4f, %.4f, %.4f' %
                            ((epoch+1)*eval_epoch, idx + 1, recon_loss.item(),inpaint_loss.item(),init_loss.item()))
                if method =='sep_joint_train':   
                    print('[%3d, %3d] loss: %.4f, %.4f' %
                            ((epoch+1)*eval_epoch, idx + 1, (1-torch.mean(recon_loss)).item(),(1-torch.mean(dsc_loss)).item()))          
        
        print("Ready validation")
        # epoch 4 weird
        # validation
        if (epoch+1) % 1 == 0 or test_only:
            print("Start evaluation")
            model.eval()
            score = {}
            if method =='vae_train':
                dsc_pancreas = 0.0
                with torch.no_grad():  
                    for val_idx,val_batch in enumerate(val_loader):
                        val_batch[label_key+'_only'] = val_batch[label_key].type(torch.cuda.LongTensor)
                        one_hot = torch.cuda.FloatTensor(val_batch[label_key+'_only'].size(0),len(mask_index),val_batch[label_key+'_only'].size(2),val_batch[label_key+'_only'].size(3),val_batch[label_key+'_only'].size(4)).zero_()
                        val_batch[label_key+'_only'] = one_hot.scatter_(1,val_batch[label_key+'_only'].data,1)
                        val_batch[label_key+'_recon'],_,_ = model(val_batch[label_key+'_only'],if_random=False)
                        if save_more_reference and val_idx == epoch % len(val_loader):
                            h=val_batch[label_key+'_only'][0:1,0:1,:,:,:].shape[4]
                            val_batch[label_key+'_display']= torch.cat((val_batch[label_key+'_only'][0:1,0:1,:,:,h//2], \
                                    val_batch[label_key+'_only'][0:1,1:2,:,:,h//2],val_batch[label_key+'_recon'][0:1,1:2,:,:,h//2]), dim=0)
                        score[val_idx] = avg_dsc(val_batch,source_key=label_key+'_recon', target_key=label_key+'_only',binary=True,botindex=1,topindex=len(mask_index)).item()
                        dsc_pancreas += score[val_idx]
                        
                    dsc_pancreas /= (val_idx+1)
            
            if method =='seg_train' or method =='joint_train' or method == 'domain_adaptation' or method =='sep_joint_train' or method =='embed_train' or method =='refine_vae':
                dsc_pancreas = 0.0
                display_image = {}
                with torch.no_grad():  
                    for val_idx,val_batch in enumerate(val_loader):
                        val_batch[label_key+'_only'] = val_batch[label_key].type(torch.cuda.LongTensor)
                        one_hot = torch.cuda.FloatTensor(val_batch[label_key+'_only'].size(0),len(mask_index),val_batch[label_key+'_only'].size(2),val_batch[label_key+'_only'].size(3),val_batch[label_key+'_only'].size(4)).zero_()
                        val_batch[label_key+'_only'] = one_hot.scatter_(1,val_batch[label_key+'_only'].data,1)
                        val_batch[img_key] = val_batch[img_key].cuda()
                        if method=='joint_train' or method == 'domain_adaptation' or method=='sep_joint_train':
                            val_batch = model(val_batch,img_key,label_key+'_pred',label_key+'_recon_pred')

                            if save_eval_result and epoch % 10 == 0:
                                filename = os.path.join(result_path, f'{epoch}_{val_idx}_pred.join')
                                np.save(filename, binarize(val_batch[label_key+'_pred']).cpu().detach().numpy())
                                filename = os.path.join(result_path, f'{epoch}_{val_idx}_pic')
                                np.save(filename, val_batch[img_key].cpu().detach().numpy())
                                filename = os.path.join(result_path, f'{epoch}_{val_idx}_gt')
                                np.save(filename, binarize(val_batch[label_key+'_only']).cpu().detach().numpy())
                            if save_more_reference and val_idx == epoch % len(val_loader):
                                h=val_batch[label_key+'_pred'][0:1,0:1,:,:,:].shape[4]
                                # if method == 'domain_adaptation':
                                #     filename = os.path.join(middle_path, f'{val_idx}_pred.npy')
                                #     with open(filename, 'rb') as f:
                                #         dat = np.load(f)
                                #     val_batch[label_key+'_display']= torch.cat((val_batch[label_key+'_recon_pred'][0:1,1:2,:,:,h//2], \
                                #         val_batch[label_key+'_only'][0:1,1:2,:,:,h//2],val_batch[label_key+'_pred'][0:1,1:2,:,:,h//2], \
                                #         dat), dim=0)
                                # else:
                                val_batch[label_key+'_display']= torch.cat((val_batch[label_key+'_recon_pred'][0:1,1:2,:,:,h//2], \
                                    val_batch[label_key+'_only'][0:1,1:2,:,:,h//2],val_batch[label_key+'_pred'][0:1,1:2,:,:,h//2]), dim=0)
                                display_image.update({label_key+'_display_val':val_batch[label_key+'_display']})
                        elif method=='embed_train' or method=='refine_vae':
                            val_batch = model(val_batch,img_key,label_key+'_pred',test_mode=True)
                        else:
                            val_batch = model(val_batch,img_key,label_key+'_pred')
                            # if save_eval_result and epoch % 10 == 0:
                            #     # filename = os.path.join(result_path, f'{epoch}_{val_idx}_pred.npy')
                            #     # np.save(filename, binarize(val_batch[label_key+'_pred']).detach().cpu().numpy())
                            #     filename = os.path.join(result_path, f'{epoch}_{val_idx}_gt.npy')
                            #     np.save(filename, binarize(val_batch[label_key+'_only']).detach().cpu().numpy())

                            if save_eval_result and epoch % 10 == 0:
                                filename = os.path.join(result_path, f'{epoch}_{val_idx}_pred.join')
                                np.save(filename, binarize(val_batch[label_key+'_pred']).cpu().detach().numpy())
                                filename = os.path.join(result_path, f'{epoch}_{val_idx}_pic')
                                np.save(filename, val_batch[img_key].cpu().detach().numpy())
                                filename = os.path.join(result_path, f'{epoch}_{val_idx}_gt')
                                np.save(filename, binarize(val_batch[label_key+'_only']).cpu().detach().numpy())
                                if load_prefix_vae is not None:
                                    val_batch[label_key+'_only_recon'],_,_ = model_ref(val_batch[label_key+'_only'],if_random=False,scale=0)
                                    filename = os.path.join(result_path, f'{epoch}_{val_idx}_gt_recon')
                                    np.save(filename, binarize(val_batch[label_key+'_only_recon']).cpu().detach().numpy())
                            if save_more_reference and val_idx == epoch % len(val_loader) and load_prefix_vae is not None:
                                val_batch[label_key+'_recon_pred'],_,_ = model_ref(val_batch[label_key+'_pred'],if_random=False,scale=0)
                                h = val_batch[label_key+'_pred'][0:1,0:1,:,:,:].shape[4]
                                val_batch[label_key+'_display']= torch.cat((val_batch[img_key][0:1,0:1,:,:,h//2], \
                                        val_batch[label_key+'_only'][0:1,1:2,:,:,h//2], val_batch[label_key+'_pred'][0:1,1:2,:,:,h//2], \
                                        val_batch[label_key+'_recon_pred'][0:1,1:2,:,:,h//2]), dim=0)
                                        
                                display_image.update({label_key+'_display_val':val_batch[label_key+'_display']})
                        
                        score[val_idx] = avg_dsc(val_batch,source_key=label_key+'_pred', target_key=label_key+'_only',binary=True,botindex=1,topindex=len(mask_index)).item()
                        dsc_pancreas += score[val_idx]

                    dsc_pancreas /= (val_idx+1)

                    if not test_only and save_more_reference:
                        for tr_idx,tr_batch in enumerate(train_loader_2):
                            if tr_idx != epoch % len(train_loader_2): continue
                            tr_batch[label_key+'_only'] = tr_batch[label_key].type(torch.cuda.LongTensor)
                            one_hot = torch.cuda.FloatTensor(tr_batch[label_key+'_only'].size(0),len(mask_index),tr_batch[label_key+'_only'].size(2),tr_batch[label_key+'_only'].size(3),tr_batch[label_key+'_only'].size(4)).zero_()
                            tr_batch[label_key+'_only'] = one_hot.scatter_(1,tr_batch[label_key+'_only'].data,1)
                            tr_batch[img_key] = tr_batch[img_key].cuda()
                            if method=='joint_train' or method == 'domain_adaptation' or method=='sep_joint_train':
                                tr_batch = model(tr_batch,img_key,label_key+'_pred',label_key+'_recon_pred')
                                h=tr_batch[label_key+'_pred'][0:1,0:1,:,:,:].shape[4]
                                tr_batch[label_key+'_display']= torch.cat((tr_batch[label_key+'_recon_pred'][0:1,1:2,:,:,h//2], \
                                        tr_batch[label_key+'_only'][0:1,1:2,:,:,h//2],tr_batch[label_key+'_pred'][0:1,1:2,:,:,h//2]), dim=0)
                            elif method=='embed_train' or method=='refine_vae':
                                tr_batch = model(tr_batch,img_key,label_key+'_pred',test_mode=True)
                            else:
                                tr_batch = model(tr_batch,img_key,label_key+'_pred')
                                h = tr_batch[label_key+'_pred'][0:1,0:1,:,:,:].shape[4]
                                if load_prefix_vae != None:
                                    tr_batch[label_key+'_recon_pred'],_,_ = model_ref(tr_batch[label_key+'_pred'],if_random=False,scale=0)
                                    if method == 'domain_adaptation':
                                        filename = os.path.join(middle_path, f'{tr_idx}_pred.pt')
                                        dat = torch.load(filename)
                                        tr_batch[label_key+'_display']= torch.cat((tr_batch[img_key][0:1,0:1,:,:,h//2], \
                                            tr_batch[label_key+'_only'][0:1,1:2,:,:,h//2], tr_batch[label_key+'_pred'][0:1,1:2,:,:,h//2], \
                                            tr_batch[label_key+'_recon_pred'][0:1,1:2,:,:,h//2], \
                                            dat[0:1,1:2,:,:,h//2]), dim=0)
                                    else:
                                        tr_batch[label_key+'_display']= torch.cat((tr_batch[img_key][0:1,0:1,:,:,h//2], \
                                            tr_batch[label_key+'_only'][0:1,1:2,:,:,h//2], tr_batch[label_key+'_pred'][0:1,1:2,:,:,h//2], \
                                            tr_batch[label_key+'_recon_pred'][0:1,1:2,:,:,h//2]), dim=0)
                                else:
                                    tr_batch[label_key+'_display']= torch.cat((tr_batch[img_key][0:1,0:1,:,:,h//2], \
                                        tr_batch[label_key+'_only'][0:1,1:2,:,:,h//2], tr_batch[label_key+'_pred'][0:1,1:2,:,:,h//2]), dim=0)

                            display_image.update({label_key+'_display_train':tr_batch[label_key+'_display']})

            output_score = os.path.join(display_path, f"score_{epoch}.json")
            with open(output_score, "w") as f:
                json.dump(score, f)

            loss = []
            loss.append(['val_result', dsc_pancreas])
            saver.write_display((epoch+1)*(max_idx_in_epoch+1), loss, display_image, force_write=True)

            print('epoch %d validation result: %f, best result %f.' % (epoch+1, dsc_pancreas, best_result))
            if test_only: break
            model.train()
            if method=='joint_train' or method=='sep_joint_train' or method=='domain_adaptation':
                model.Vae.eval()
        
        ## save model
        if (epoch+1) % (save_epoch // eval_epoch) == 0:
            if not os.path.exists(save_path):
                save_path = '3dmodel/'+prefix
                os.mkdir(save_path)
            print('saving model')
            torch.save({
                        'epoch': (epoch+1)*eval_epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                        }, os.path.join(save_path,'model_epoch'+str((epoch+1)*eval_epoch)+'.ckpt'))
            if dsc_pancreas > best_result:
                best_result = dsc_pancreas
                torch.save({
                        'epoch': (epoch+1)*eval_epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                        }, os.path.join(save_path,'best_model.ckpt'))
            '''
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': generator_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                        }, os.path.join(save_path,'generator_model_epoch'+str(epoch+1)+'.ckpt'))
            '''
    print('Finished Training')


