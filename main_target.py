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
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time


from scipy.ndimage.morphology import binary_dilation, binary_erosion
from utils.utils import plot_slides,BaseDataset,NumpyLoader_Multi, NumpyLoader_Multi_merge,NiiLoader, image_resize,CropResize,CopyField, ExtendSqueeze,Reshape, PadToSize, Clip, Binarize, CenterIntensities, get_parameter_number
from utils.evaluation import dice, avg_ce, KLloss, avg_dsc, binarize, confident_binarize
from utils.saver import Saver
from utils.draw import scatter_plot, scatter_plot_multi
import random
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
parser.add_argument("--data_root", help="data_root", default='../nih_data/numpy_data/')
parser.add_argument("--val_data_root", help="val_data_root", default='../nih_data/numpy_data/')
parser.add_argument("--pseudo_data_root", help="pseudo_data_root", default='../nih_data/numpy_data/')
parser.add_argument("-l","--data_path", help="data_path", default='Multi_all.json')
parser.add_argument("--train_list", help="train_list", default='NIH_train')
parser.add_argument("--val_list", help="val_list", default='NIH_val')
parser.add_argument("--pseudo_list", help="pseudo_list", default=None)
parser.add_argument("--load_prefix", help="load_prefix", default=None)
parser.add_argument("--checkpoint_name", help="checkpoint_name", default="best_model.ckpt")
parser.add_argument("--load_prefix_vae", help="load_prefix_vae", default=None)
parser.add_argument("--load_prefix_encoder", help="load_prefix_encoder", default=None)
parser.add_argument("--load_prefix_joint", help="load_prefix_joint", default=None)
parser.add_argument("--pan_index", help="pan_index", default='1')
parser.add_argument("--pseudo_pan_index", help="pseudo_pan_index", default='1')
parser.add_argument("--lambda_vae", type=float, help="lambda_vae", default=0.1)
parser.add_argument("--lambda_vae_warmup", type=int, help="save_epoch",default=0)
parser.add_argument("--lr_seg", type=float, help="lr_seg", default=1e-2) # for seg 1e-1
parser.add_argument("--lr_vae", type=float, help="lr_vae", default=0) # for vae 1e-1
parser.add_argument("--test_only", help="test_only", action='store_true')
parser.add_argument("--resume", help="resume", action='store_true')
parser.add_argument("--save_more_reference", help="save_more_reference", action='store_true')
parser.add_argument("--save_eval_result", help="save_more_reference", action='store_true')
parser.add_argument("--no_aug", help="no_aug", action='store_true')
parser.add_argument("--only_pseudo", help="only_pseudo", action='store_true')
parser.add_argument("--fix_layer", help="fix_layer", action='store_true')
parser.add_argument("--use_confident_binarize", help="confident_binarize", action='store_true')
parser.add_argument("--analysis_figure_name", help="analysis_figure_name", default=None)
parser.add_argument("--pseudo_save_epoch", help="pseudo_save_epoch", type=int, default=0)
parser.add_argument("--domain_loss_type", type=int, help="domain_loss_type",default=0)
parser.add_argument("--vae_mont_number", type=int, help="number of times to run the vae network", default=1)
parser.add_argument("--vae_forward_scale", type=float, help="vae_forward_scale",default=0.0)
parser.add_argument("--vae_decoder_dropout", type=float, help="vae_decoder_dropout",default=0.0)
parser.add_argument("--seg_dropout", type=float, help="seg_dropout",default=0.0)
parser.add_argument("--val_finetune", type=int, help="val_finetune", default=0)
parser.add_argument("--lr_finetune", type=float, help="lr_finetune", default=1e-2) # for seg 1e-1
parser.add_argument("--tag", help="tag", action='store_true')
parser.add_argument("--from_scratch", help="Use only in domain adaptation. Train from scratch in the target domain.", action='store_true')
parser.add_argument("--adam", help="adam", action='store_true')
parser.add_argument("--kl", help="use kl loss in domain adaptation", action='store_true')
parser.add_argument("--alpha", type=float, help="alpha", default=0.995)
parser.add_argument("--update_every_iteration", action='store_true')
parser.add_argument("--generate_bounding_boxes", action='store_true')
parser.add_argument("--shift", type=int, default=0)
args = parser.parse_args()

data_root = args.data_root
val_data_root = args.val_data_root
pseudo_data_root = args.pseudo_data_root
lr1 = args.lr_seg
lr2 = args.lr_vae
softrelu = args.softrelu
train_list = args.train_list
val_list = args.val_list
pseudo_list = args.pseudo_list
torch.backends.cudnn.benchmark = True
weight_decay = 0
num_workers = 16
trainbatch = args.batch_size
valbatch = 1
load_prefix = args.load_prefix
checkpoint_name = args.checkpoint_name
load_prefix_vae = args.load_prefix_vae
load_prefix_encoder = args.load_prefix_encoder
load_prefix_joint = args.load_prefix_joint
load_epoch_seg = 240
load_epoch = 60
prefix = args.prefix
data_path = os.path.join('lists',args.data_path)
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
save_root_path = '3dmodel'
save_path = '3dmodel/'+prefix
if not os.path.exists(save_path): os.mkdir(save_path)
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
pseudo_pan_index = args.pseudo_pan_index
if pseudo_pan_index != '10':
    pseudo_mask_index = [[0,0]] + [[int(f), idx+1] for idx,f in enumerate(pseudo_pan_index.split(','))]
else:
    pseudo_mask_index = [[0,0], [[1,2], 1]]
target_phase = args.target_phase
lambda_vae = args.lambda_vae
lambda_vae_warmup = args.lambda_vae_warmup
test_only = args.test_only
resume = args.resume
method = args.method
save_more_reference = args.save_more_reference
save_eval_result = args.save_eval_result
only_pseudo = args.only_pseudo
fix_layer = args.fix_layer
use_confident_binarize = args.use_confident_binarize
analysis_figure_name = args.analysis_figure_name
domain_loss_type = args.domain_loss_type
vae_mont_number = args.vae_mont_number
vae_forward_scale = args.vae_forward_scale
if vae_mont_number != 1: assert vae_forward_scale != 0.0
vae_decoder_dropout = args.vae_decoder_dropout
seg_dropout = args.seg_dropout
val_finetune = args.val_finetune
lr_finetune = args.lr_finetune
if save_eval_result and not os.path.exists(result_path):
    os.mkdir(result_path)
if analysis_figure_name != None:
    assert test_only
no_aug = args.no_aug
tag = args.tag
from_scratch = args.from_scratch
if from_scratch: 
    assert method == 'domain_adaptation'
    assert not test_only
adam = args.adam
kl = args.kl
if kl: 
    assert method == 'domain_adaptation'
    assert domain_loss_type == 0 or domain_loss_type == 8
alpha = args.alpha
update_every_iteration = args.update_every_iteration
pseudo_save_epoch = args.pseudo_save_epoch
if update_every_iteration: assert pseudo_save_epoch == 1
generate_bounding_boxes = args.generate_bounding_boxes
if generate_bounding_boxes: assert method == 'domain_adaptation'
shift = args.shift


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

if __name__ == "__main__":
    ## dataset
    train_data_list = filedict_from_json(data_path, train_list, eval_epoch)

    transforms = {'train': []}
    ## define training data pipeline
    transforms['train'].append(NumpyLoader_Multi_merge(fields=input_phases, root_dir=data_root, load_mask=True, mask_index=mask_index))
    transforms['train'].append(CropResize(fields=input_phases, output_size=patch_size, shift=shift))
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

    ## dataset
    if pseudo_list is not None:
        pseudo_data_list = filedict_from_json(data_path, pseudo_list, eval_epoch)
        # print(train_data_list)

        transforms['pseudo']=[]
        transforms['pseudo'].append(NumpyLoader_Multi_merge(fields=input_phases, root_dir=pseudo_data_root, load_mask=True, mask_index=pseudo_mask_index))
        transforms['pseudo'].append(CropResize(fields=input_phases, output_size=patch_size))
        #transforms['train'].append(PadToSize(fields=input_phases, size=input_size, pad_val=-1024, seg_pad_val=0,random_subpadding=True,load_mask=method=='seg_train'))
        
        if not no_aug:
            transforms['pseudo'].append(Reshape(fields=input_phases_mask))
            transforms['pseudo'].append(MySpatialTransform(patch_size,[dis//2-5 for dis in patch_size], random_crop=True,
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
            transforms['pseudo'].append(CopyField(fields=[phase], to_field=[phase+'_origin']))
        transforms['pseudo'].append(Clip(fields=input_phases,new_min=-200, new_max=400))
        transforms['pseudo'].append(CenterIntensities(fields=input_phases,subtrahend=100, divisor=300))
        transforms['pseudo'].append(Reshape(fields=input_phases_mask, reshape_view=[-1]+patch_size))

    val_data_list = filedict_from_json(data_path, val_list)
    transforms['val']=[]
    ## define validation data pipeline
    transforms['val'].append(NumpyLoader_Multi_merge(fields=input_phases, root_dir=val_data_root,load_mask=True,mask_index=mask_index))
    transforms['val'].append(CropResize(fields=input_phases,output_size=patch_size, shift=shift))
    #transforms['val'].append(PadToSize(fields=input_phases, size=input_size, pad_val=-1024, seg_pad_val=0,random_subpadding=False,load_mask=True))
    for phase in input_phases:
        transforms['val'].append(CopyField(fields=[phase], to_field=[phase+'_origin']))
    transforms['val'].append(Clip(fields=input_phases,new_min=-200, new_max=400))
    transforms['val'].append(CenterIntensities(fields=input_phases,subtrahend=100, divisor=300))
    transforms['val'].append(Reshape(fields=input_phases_mask, reshape_view=[-1]+patch_size))

    if method == "discriminator_train":
        transforms = {'train': []}
        ## define training data pipeline
        transforms['train'].append(NumpyLoader_Multi_merge(fields=input_phases, root_dir=data_root, load_seg_npy=True, mask_index=mask_index))
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
        transforms['train'].append(Reshape(fields=input_phases_mask, reshape_view=[-1]+patch_size))

        val_data_list = filedict_from_json(data_path, val_list)
        transforms['val']=[]
        ## define validation data pipeline
        transforms['val'].append(NumpyLoader_Multi_merge(fields=input_phases, root_dir=val_data_root,load_seg_npy=True,mask_index=mask_index))
        transforms['val'].append(Reshape(fields=input_phases_mask, reshape_view=[-1]+patch_size))


    for k,v in transforms.items():
        transforms[k] = torchvision.transforms.Compose(v)

    ###############################################################################################
    ############################ Create Datasets ##################################################
    ###############################################################################################
    print("Loading data.")
    train_dataset = BaseDataset(train_data_list, transforms=transforms['train'])
    val_dataset = BaseDataset(val_data_list, transforms=transforms['val'])
    if pseudo_list is not None:
        pseudo_dataset = BaseDataset(pseudo_data_list, transforms=transforms['pseudo'])
    train_loader = DataLoader(train_dataset, batch_size=trainbatch, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=valbatch, shuffle=False, num_workers=num_workers, pin_memory=True)
    if pseudo_list is not None:
        pseudo_loader = DataLoader(pseudo_dataset, batch_size=trainbatch, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)
    if save_more_reference:
        train_loader_2 = DataLoader(train_dataset, batch_size=valbatch, shuffle=False, num_workers=num_workers, pin_memory=True)

    ## model build and load
    print("Building model.")
    models = importlib.import_module('joint_model')
    # vm_model = importlib.import_module('models.' + 'voxelmorph3D_joint')
    if method=='vae_train':
        model=models.VAE(n_channels=len(mask_index), n_class=len(mask_index),norm_type=1,dim=128,soft=softrelu==1)
    elif method == 'discriminator_train':
        model = models.Encoder(n_channels=1, dim=1, norm_type=1)
    elif method == 'domain_adaptation':
        model=[]
        model.append(models.Segmentation(n_channels=1, n_class=len(mask_index),norm_type=1))
        model.append(models.VAE(n_channels=len(mask_index), n_class=len(mask_index),norm_type=1,dim=128))
        model = models.Joint(models=model, vae_forward_scale=vae_forward_scale, vae_decoder_dropout=vae_decoder_dropout)
        model_fix=[]
        model_fix.append(models.Segmentation(n_channels=1, n_class=len(mask_index),norm_type=1))
        model_fix.append(models.VAE(n_channels=len(mask_index), n_class=len(mask_index),norm_type=1,dim=128))
        model_fix = models.Joint(models=model_fix)
        if val_finetune != 0:
            model_finetune=[]
            model_finetune.append(models.Segmentation(n_channels=1, n_class=len(mask_index),norm_type=1))
            model_finetune.append(models.VAE(n_channels=len(mask_index), n_class=len(mask_index),norm_type=1,dim=128))
            model_finetune = models.Joint(models=model_finetune)
            for param in model_finetune.Vae.parameters():
                param.requires_grad = False
            model_finetune.Vae.eval()
    elif method == 'domain_adaptation_dis':
        model=[]
        model.append(models.Segmentation(n_channels=1, n_class=len(mask_index),norm_type=1))
        model.append(models.Encoder(n_channels=1, dim=1, norm_type=1))
        model = models.Joint2(models=model)
        model_fix = models.Segmentation(n_channels=1, n_class=len(mask_index),norm_type=1)
    else:
        raise ValueError("Try a valid method.")
    model = model.cuda()

    if adam:
        optimizer = torch.optim.Adam(model.parameters(),
                                lr=lr1,betas=(0.9, 0.999),weight_decay = weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=lr1,weight_decay = weight_decay,momentum=0.9)
    
    print("Loading prefix.")
    if load_prefix:
        register_model_path = os.path.join(save_root_path, load_prefix, checkpoint_name)
        if method=="seg_train":
            model.load_state_dict(torch.load(register_model_path)['model_state_dict'])
        else:
            if from_scratch:
                model_fix.Seg.load_state_dict(torch.load(register_model_path)['model_state_dict'])
            else:
                model.Seg.load_state_dict(torch.load(register_model_path)['model_state_dict'])
            if method == 'domain_adaptation_dis':
                model_fix.load_state_dict(model.Seg.state_dict())

    if load_prefix_vae:
        register_model_path = save_root_path+'/'+load_prefix_vae+'/best_model.ckpt'

        if from_scratch: 
            model_fix.Vae.load_state_dict(torch.load(register_model_path)['model_state_dict'])
        model.Vae.load_state_dict(torch.load(register_model_path)['model_state_dict'])
        '''
        pretrained_dict = torch.load(register_model_path)['model_state_dict']
        model_dict = register_model.segmentation_model.state_dict()
        pretrained_dict = {k.split('.',1)[1]: v for k, v in pretrained_dict.items() if k.split('.',1)[0]=='segmentation_model' and k.split('.',1)[1] in model_dict}
        model_dict.update(pretrained_dict) 
        register_model.segmentation_model.load_state_dict(model_dict)
        register_model.segmentation_model.eval()
        '''
    if method == "domain_adaptation" and test_only:
        model_fix.load_state_dict(model.state_dict())
    
    if load_prefix_encoder:
        register_model_path = save_root_path+'/'+load_prefix_encoder+'/best_model.ckpt'
        if method == 'discriminator_train':
            model.load_state_dict(torch.load(register_model_path)['model_state_dict'])
        else:
            print("Loading domain-dis discriminator.")
            model.Dis.load_state_dict(torch.load(register_model_path)['model_state_dict'])

    if load_prefix_joint:
        register_model_path = save_root_path+'/'+load_prefix_joint+'/best_model.ckpt'
        model.load_state_dict(torch.load(register_model_path)['model_state_dict'])
    
    if method=='joint_train' or method=='domain_adaptation':
        for param in model.Vae.parameters():
            param.requires_grad = False
        model.Vae.eval()
        if fix_layer:
            for param in model.Seg.parameters():
                param.requires_grad = False
            for param in model.Seg.up5.parameters():
                param.requires_grad = True
            for param in model.Seg.out_block.parameters():
                param.requires_grad = True

    if method == 'domain_adaptation_dis':
        for param in model.Dis.parameters():
            param.requires_grad = False
        model.Dis.eval()

    if method=='refine_vae':
        Encoder_list = ['in_block','down1','down2','down3','down4','down5','fc_mean','fc_std']
        for param_name,param in model.Vae.named_parameters():
            if param_name.split('.')[0] in Encoder_list:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    if method == 'domain_adaptation':
        if only_pseudo:
            tmp = model_fix
            model_fix = model
            model = tmp
        else:
            if not test_only and not from_scratch:
                model_fix.load_state_dict(model.state_dict())

    if method == 'domain_adaptation' or method == 'domain_adaptation_dis':
        for param in model_fix.parameters():
            param.requires_grad = False
        model_fix.eval()

    
    final_model_parallel = nn.DataParallel(model).cuda()
    if method == 'domain_adaptation' or method == 'domain_adaptation_dis': model_fix_parallel = nn.DataParallel(model_fix).cuda()
    if val_finetune != 0: model_finetune_parallel = nn.DataParallel(model_finetune).cuda()
    label_key = 'venous_pancreas'
    img_key = 'venous'
    best_result = 0
    train_dis = 0
    max_idx_in_epoch = 0
    saver = Saver(display_dir=display_path,display_freq=10)
    MSE_Loss = MSELoss()
    

    # get_parameter_number(model)
    # get_parameter_number(model.Seg)
    # get_parameter_number(model.Vae)
    # get_parameter_number(model_fix)



    ## training loop 
    print("Start training")
    if pseudo_list is not None: pseudo_itr = iter(pseudo_loader)
    for epoch in range(max_epoch // eval_epoch):
        if not test_only:
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

                if method == 'discriminator_train':
                    batch[label_key+'_only'] = batch[label_key].type(torch.cuda.FloatTensor)
                    score = batch['venous_score'].type(torch.cuda.FloatTensor)
                    score_out = final_model_parallel(batch[label_key+'_only'])
                    final_loss = torch.square(score - score_out).mean()
                    # import pdb; pdb.set_trace()
                    loss = []
                    loss.append(['final_loss',final_loss.item()])
                    loss.append(['score_out',score_out[0].item()])
                    saver.write_display(idx+epoch*(max_idx_in_epoch+1),loss)

                if method =='domain_adaptation' and pseudo_list is None:
                    if epoch == 0: continue

                    if pseudo_save_epoch != 0 and epoch % (pseudo_save_epoch // eval_epoch) == 0:
                        if idx % (len(train_loader) / eval_epoch) == 0 or update_every_iteration:
                            if not update_every_iteration: print("Updating Network")
                            # model_fix.load_state_dict(model.state_dict())
                            sd_teacher = model_fix.Seg.state_dict()
                            sd_student = model.Seg.state_dict()
                            for key in sd_student:
                                sd_teacher[key] = alpha * sd_teacher[key] + (1 - alpha) * sd_student[key]
                            model_fix.Seg.load_state_dict(sd_teacher)
                            # if tag: lambda_vae /= 10
                            if tag: lambda_vae = alpha * lambda_vae

                    batch[label_key+'_only'] = batch[label_key].type(torch.cuda.LongTensor)
                    one_hot = torch.cuda.FloatTensor(batch[label_key+'_only'].size(0),len(mask_index),batch[label_key+'_only'].size(2),batch[label_key+'_only'].size(3),batch[label_key+'_only'].size(4)).zero_()
                    batch[label_key+'_only'] = one_hot.scatter_(1,batch[label_key+'_only'].data,1)
                    batch[img_key] = batch[img_key].cuda()

                    total_recon_loss = 0.0
                    total_dice_loss_fake = 0.0
                    total_dice_loss = 0.0
                    total_final_loss = 0.0

                    for i in range(vae_mont_number):
                        batch = final_model_parallel(batch,img_key,label_key+'_pred',label_key+'_recon_pred',dropout=True)
                        batch = model_fix_parallel(batch,img_key,label_key+'_only_fake',label_key+'_asdfasdf')
                        h=batch[label_key+'_pred'][0:1,0:1,:,:,:].shape[4]
                        if use_confident_binarize:
                            batch[label_key+'_only_fake'] = confident_binarize(batch[label_key+'_only_fake'])
                        else:
                            batch[label_key+'_only_fake'] = binarize(batch[label_key+'_only_fake'])

                        batch[label_key+'_display']= torch.cat((batch[label_key+'_recon_pred'][0:1,1:2,:,:,h//2], \
                                batch[label_key+'_only'][0:1,1:2,:,:,h//2],batch[label_key+'_pred'][0:1,1:2,:,:,h//2], \
                                batch[label_key+'_only_fake'][0:1,1:2,:,:,h//2]), dim=0)
                        
                        recon_loss = 1 - avg_dsc(batch,source_key=label_key+'_pred', target_key=label_key+'_recon_pred',botindex=1,topindex=len(mask_index),return_mean=True)
                        klloss = KLloss(batch)
                        dsc_loss = 1 - avg_dsc(batch,source_key=label_key+'_pred', target_key=label_key+'_only',botindex=1,topindex=len(mask_index),return_mean=True)
                        dsc_loss_fake = 1 - avg_dsc(batch,source_key=label_key+'_pred', target_key=label_key+'_only_fake',botindex=1,topindex=len(mask_index),return_mean=True)

                        if only_pseudo:
                            final_loss = dsc_loss_fake
                        elif domain_loss_type == 8 or domain_loss_type == 15 or domain_loss_type == 16:
                            if recon_loss < 0.15: cur_lambda = lambda_vae * 0.6
                            elif recon_loss < 0.225: cur_lambda = lambda_vae * 1.2
                            elif recon_loss < 0.3: cur_lambda = lambda_vae * 2.0
                            else: cur_lambda = lambda_vae * 3.0
                            if cur_lambda > 1:
                                if kl: final_loss = recon_loss + klloss + 1 / cur_lambda * dsc_loss_fake
                                else: final_loss = recon_loss + 1 / cur_lambda * dsc_loss_fake
                            else:
                                if kl: final_loss = cur_lambda * (recon_loss + klloss) + dsc_loss_fake
                                else: final_loss = cur_lambda * recon_loss + dsc_loss_fake
                        elif domain_loss_type == 9:
                            if recon_loss < 0.15: cur_lambda = lambda_vae * 0.6
                            elif recon_loss < 0.225: cur_lambda = lambda_vae * 1.2
                            elif recon_loss < 0.3: cur_lambda = lambda_vae * 2.0
                            else: cur_lambda = lambda_vae * 3.0
                            final_loss = (cur_lambda * recon_loss + dsc_loss_fake) / (1 + cur_lambda)
                        elif domain_loss_type == 10:
                            loss_square = torch.mean(torch.square(val_batch[label_key+'_pred'])) 
                            final_loss = loss_square + recon_loss + dsc_loss_fake
                            print(final_loss)
                        elif domain_loss_type == 11:
                            final_loss = lambda_vae * recon_loss + dsc_loss_fake + recon_loss * dsc_loss_fake
                        elif domain_loss_type == 12:
                            final_loss = lambda_vae * recon_loss + dsc_loss_fake - recon_loss * dsc_loss_fake
                        elif domain_loss_type == 13:
                            recon_loss -= 0.15
                            recon_loss[recon_loss < 0] = 0
                            final_loss = lambda_vae * recon_loss
                        elif domain_loss_type == 14:
                            recon_loss -= 0.1
                            recon_loss[recon_loss < 0] = 0
                            final_loss = lambda_vae * recon_loss + dsc_loss_fake
                        elif turn_epoch != -1:
                            if (epoch // turn_epoch) % 2 == 0:
                                final_loss = lambda_vae * recon_loss
                            else:
                                final_loss = lambda_vae * recon_loss + dsc_loss_fake
                        elif epoch >= lambda_vae_warmup:
                            final_loss = lambda_vae * recon_loss + dsc_loss_fake
                            if kl: final_loss += 0.00002 * lambda_vae * klloss
                        else:
                            final_loss = lambda_vae * epoch / lambda_vae_warmup * recon_loss + dsc_loss_fake
                            

                        total_recon_loss += recon_loss
                        total_dice_loss_fake += dsc_loss_fake
                        total_dice_loss += dsc_loss
                        total_final_loss += final_loss

                    recon_loss = total_recon_loss / vae_mont_number
                    dsc_loss_fake = total_dice_loss_fake / vae_mont_number
                    dsc_loss = total_dice_loss / vae_mont_number
                    final_loss = total_final_loss / vae_mont_number
                    loss = []
                    display_image={}
                    loss.append(['recon_loss',recon_loss.item()])
                    loss.append(['kl_loss', klloss.item()])
                    loss.append(['dice_loss_fake',dsc_loss_fake.item()])
                    loss.append(['dice_loss',dsc_loss.item()])
                    loss.append(['final_loss',final_loss.item()])
                    loss.append(['lr',optimizer.param_groups[0]['lr']])
                    display_image.update({label_key+'_display':batch[label_key+'_display']})
                    saver.write_display(idx+epoch*(max_idx_in_epoch+1),loss,display_image)

                if method =='domain_adaptation' and pseudo_list is not None:
                    if epoch == 0: continue
                    batch[label_key+'_only'] = batch[label_key].type(torch.cuda.LongTensor)
                    one_hot = torch.cuda.FloatTensor(batch[label_key+'_only'].size(0),len(mask_index),batch[label_key+'_only'].size(2),batch[label_key+'_only'].size(3),batch[label_key+'_only'].size(4)).zero_()
                    batch[label_key+'_only'] = one_hot.scatter_(1,batch[label_key+'_only'].data,1)
                    batch[img_key] = batch[img_key].cuda()
                    batch = final_model_parallel(batch,img_key,label_key+'_pred',label_key+'_recon_pred',dropout=True)
                    batch = model_fix_parallel(batch,img_key,label_key+'_only_fake',label_key+'_asdfasdf')
                    h=batch[label_key+'_pred'][0:1,0:1,:,:,:].shape[4]
                    if use_confident_binarize:
                        batch[label_key+'_only_fake'] = confident_binarize(batch[label_key+'_only_fake'])
                    else:
                        batch[label_key+'_only_fake'] = binarize(batch[label_key+'_only_fake'])

                    batch[label_key+'_display']= torch.cat((batch[label_key+'_recon_pred'][0:1,1:2,:,:,h//2], \
                            batch[label_key+'_only'][0:1,1:2,:,:,h//2],batch[label_key+'_pred'][0:1,1:2,:,:,h//2], \
                            batch[label_key+'_only_fake'][0:1,1:2,:,:,h//2]), dim=0)

                    if pseudo_save_epoch != 0 and epoch % pseudo_save_epoch == 0:
                        model_fix.load_state_dict(model.state_dict())
                        if tag: lambda_vae /= 10
                    
                    recon_loss = 1 - avg_dsc(batch,source_key=label_key+'_pred', target_key=label_key+'_recon_pred',botindex=1,topindex=len(mask_index),return_mean=True)
                    dsc_loss = 1 - avg_dsc(batch,source_key=label_key+'_pred', target_key=label_key+'_only',botindex=1,topindex=len(mask_index),return_mean=True)
                    dsc_loss_fake = 1 - avg_dsc(batch,source_key=label_key+'_pred', target_key=label_key+'_only_fake',botindex=1,topindex=len(mask_index),return_mean=True)

                    if domain_loss_type == 8:
                        if recon_loss < 0.15: cur_lambda = lambda_vae * 0.6
                        elif recon_loss < 0.225: cur_lambda = lambda_vae * 1.2
                        elif recon_loss < 0.3: cur_lambda = lambda_vae * 2.0
                        else: cur_lambda = lambda_vae * 3.0
                        if cur_lambda > 1:
                            final_loss = recon_loss + 1 / cur_lambda * dsc_loss_fake
                        else:
                            final_loss = cur_lambda * recon_loss + dsc_loss_fake
                    elif lambda_vae >= 1000:
                        final_loss = recon_loss * lambda_vae / 10000
                    else:
                        final_loss = lambda_vae * recon_loss + dsc_loss_fake
                    loss = []
                    display_image={}
                    loss.append(['lr',optimizer.param_groups[0]['lr']])
                    loss.append(['recon_loss',recon_loss.item()])
                    loss.append(['dice_loss_fake',dsc_loss_fake.item()])
                    loss.append(['dice_loss',dsc_loss.item()])
                    loss.append(['final_loss',final_loss.item()])
                    display_image.update({label_key+'_display':batch[label_key+'_display']})

                    optimizer.zero_grad()
                    final_loss.backward()
                    optimizer.step()

                    # --------------------------------------------------------------------------------- #
                    batch = next(pseudo_itr, None)
                    if batch is None:
                        pseudo_itr = iter(pseudo_loader)
                        batch = next(pseudo_itr)
                    batch[label_key+'_only'] = batch[label_key].type(torch.cuda.LongTensor)
                    one_hot = torch.cuda.FloatTensor(batch[label_key+'_only'].size(0),len(mask_index),batch[label_key+'_only'].size(2),batch[label_key+'_only'].size(3),batch[label_key+'_only'].size(4)).zero_()
                    batch[label_key+'_only'] = one_hot.scatter_(1,batch[label_key+'_only'].data,1)
                    batch[img_key] = batch[img_key].cuda()
                    batch = final_model_parallel(batch,img_key,label_key+'_pred',label_key+'_recon_pred',dropout=True)
                    h=batch[label_key+'_pred'][0:1,0:1,:,:,:].shape[4]

                    batch[label_key+'_display']= torch.cat((batch[label_key+'_recon_pred'][0:1,1:2,:,:,h//2], \
                            batch[label_key+'_only'][0:1,1:2,:,:,h//2],batch[label_key+'_pred'][0:1,1:2,:,:,h//2]), dim=0)
                    
                    recon_loss = 1 - avg_dsc(batch,source_key=label_key+'_pred', target_key=label_key+'_recon_pred',botindex=1,topindex=len(mask_index),return_mean=True)
                    dsc_loss = 1 - avg_dsc(batch,source_key=label_key+'_pred', target_key=label_key+'_only',botindex=1,topindex=len(mask_index),return_mean=True)

                    final_loss = dsc_loss
                    loss.append(['recon_loss_pseudo',recon_loss.item()])
                    loss.append(['dice_loss_pseudo',dsc_loss.item()])
                    loss.append(['final_loss_pseudo',final_loss.item()])
                    display_image.update({label_key+'_display_pseudo':batch[label_key+'_display']})
                    
                    saver.write_display(idx+epoch*(max_idx_in_epoch+1),loss,display_image)

                if method =='domain_adaptation_dis' and pseudo_list is None:
                    if epoch == 0: continue
                    batch[label_key+'_only'] = batch[label_key].type(torch.cuda.LongTensor)
                    one_hot = torch.cuda.FloatTensor(batch[label_key+'_only'].size(0),len(mask_index),batch[label_key+'_only'].size(2),batch[label_key+'_only'].size(3),batch[label_key+'_only'].size(4)).zero_()
                    batch[label_key+'_only'] = one_hot.scatter_(1,batch[label_key+'_only'].data,1)
                    batch[img_key] = batch[img_key].cuda()
                    batch = final_model_parallel(batch, img_key, label_key+'_pred', label_key+'_score', dropout=True)
                    batch = model_fix_parallel(batch, img_key, label_key+'_only_fake')
                    h=batch[label_key+'_pred'][0:1,0:1,:,:,:].shape[4]
                    if use_confident_binarize:
                        batch[label_key+'_only_fake'] = confident_binarize(batch[label_key+'_only_fake'])
                    else:
                        batch[label_key+'_only_fake'] = binarize(batch[label_key+'_only_fake'])

                    batch[label_key+'_display']= torch.cat((batch[label_key+'_only'][0:1,1:2,:,:,h//2], \
                            batch[label_key+'_pred'][0:1,1:2,:,:,h//2], batch[label_key+'_only_fake'][0:1,1:2,:,:,h//2]), dim=0)

                    if pseudo_save_epoch != 0 and epoch % pseudo_save_epoch == 0:
                        model_fix.load_state_dict(model.state_dict())
                        if tag: lambda_vae /= 10
                    
                    dsc_loss = 1 - avg_dsc(batch,source_key=label_key+'_pred', target_key=label_key+'_only',botindex=1,topindex=len(mask_index),return_mean=True)
                    dsc_loss_fake = 1 - avg_dsc(batch,source_key=label_key+'_pred', target_key=label_key+'_only_fake',botindex=1,topindex=len(mask_index),return_mean=True)
                    discriminator_loss = 1 - batch[label_key+'_score'].mean()

                    final_loss = lambda_vae * discriminator_loss + dsc_loss_fake

                    if epoch >= lambda_vae_warmup:
                        final_loss = lambda_vae * discriminator_loss + dsc_loss_fake
                    else:
                        final_loss = lambda_vae * epoch / lambda_vae_warmup * discriminator_loss + dsc_loss_fake
                    loss = []
                    display_image={}
                    loss.append(['discriminator_loss',discriminator_loss.item()])
                    loss.append(['dice_loss_fake',dsc_loss_fake.item()])
                    loss.append(['dice_loss',dsc_loss.item()])
                    loss.append(['final_loss',final_loss.item()])
                    loss.append(['lr',optimizer.param_groups[0]['lr']])
                    display_image.update({label_key+'_display':batch[label_key+'_display']})
                    saver.write_display(idx+epoch*(max_idx_in_epoch+1),loss,display_image)

                optimizer.zero_grad()
                final_loss.backward()
                optimizer.step()
                # print statistics
                if method =='vae_train':
                    print('[%3d, %3d] loss: %.4f, %.4f' %
                            ((epoch+1)*eval_epoch, idx + 1, dsc_loss.item(),klloss.item()))
                if method == 'discriminator_train':
                    print('[%3d, %3d] loss: %.4f, %.4f, %.4f' %
                            ((epoch+1)*eval_epoch, idx + 1, score[0], score_out[0].item(), final_loss.item()))
                if method == 'domain_adaptation':
                    print('[%3d, %3d] loss: %.4f, %.4f, %.4f' %
                            ((epoch+1)*eval_epoch, idx + 1, recon_loss.item(), dsc_loss_fake.item(), dsc_loss.item()))
                if method == 'domain_adaptation_dis':
                    print('[%3d, %3d] loss: %.4f, %.4f, %.4f' %
                            ((epoch+1)*eval_epoch, idx + 1, discriminator_loss.item(), dsc_loss_fake.item(), dsc_loss.item()))       
        
        print("Ready validation")
        # epoch 4 weird
        # validation
        if (epoch+1) % 1 == 0 or test_only:
            print("Start evaluation")
            model.eval()
            # model.Vae.eval()
            # model.train()
            # model.Vae.train()
            score = {}
            
            score_figure = {}
            score_figure_gt = {}
            score_figure_pseudo = {}
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
            
            if method == 'discriminator_train':
                dsc_pancreas = 0.0
                display_image = None
                with torch.no_grad():  
                    for val_idx,val_batch in enumerate(val_loader):
                        val_batch[label_key+'_only'] = val_batch[label_key].type(torch.cuda.FloatTensor)
                        score_real = val_batch['venous_score'].type(torch.cuda.FloatTensor)
                        score_out = final_model_parallel(val_batch[label_key+'_only'])
                        final_loss = torch.square(score_real - score_out).mean()
                        score[val_idx] = final_loss.item()
                        dsc_pancreas += 1 - final_loss.item()
                    dsc_pancreas /= (val_idx + 1)
            
            if method =='seg_train' or method =='joint_train' or method == 'domain_adaptation' or method == 'domain_adaptation_dis' or method =='sep_joint_train' or method =='embed_train' or method =='refine_vae':
                dsc_pancreas = 0.0
                if val_finetune != 0:
                    dsc_pancreas_noft = 0.0
                    score_noft = {}
                loss_gt = 0.0
                loss_recon = 0.0
                loss_fake = 0.0
                display_image = {}
                # with torch.no_grad():
                time1 = time.time()
                for val_idx,val_batch in enumerate(val_loader):
                    # if val_idx == 12: print("\n\n\n\n\n", val_batch['id'], "\n\n\n\n\n")
                    if val_finetune != 0:
                        model_finetune.load_state_dict(model.state_dict())
                        if epoch != 0 or test_only:
                            for i in range(val_finetune):
                                # batch = val_batch
                                val_batch[label_key+'_only'] = val_batch[label_key].type(torch.cuda.LongTensor)
                                one_hot = torch.cuda.FloatTensor(val_batch[label_key+'_only'].size(0),len(mask_index),val_batch[label_key+'_only'].size(2),val_batch[label_key+'_only'].size(3),val_batch[label_key+'_only'].size(4)).zero_()
                                val_batch[label_key+'_only'] = one_hot.scatter_(1,val_batch[label_key+'_only'].data,1)
                                val_batch[img_key] = val_batch[img_key].cuda()
                                val_batch = model_finetune(val_batch,img_key,label_key+'_pred',label_key+'_recon_pred',dropout=True)
                                val_batch = model_fix_parallel(val_batch,img_key,label_key+'_only_fake',label_key+'_asdfasdf')
                                klloss = KLloss(val_batch)
                                h=val_batch[label_key+'_pred'][0:1,0:1,:,:,:].shape[4]
                                if use_confident_binarize:
                                    val_batch[label_key+'_only_fake'] = confident_binarize(val_batch[label_key+'_only_fake'])
                                else:
                                    val_batch[label_key+'_only_fake'] = binarize(val_batch[label_key+'_only_fake'])

                                val_batch[label_key+'_display']= torch.cat((val_batch[label_key+'_recon_pred'][0:1,1:2,:,:,h//2], \
                                        val_batch[label_key+'_only'][0:1,1:2,:,:,h//2],val_batch[label_key+'_pred'][0:1,1:2,:,:,h//2], \
                                        val_batch[label_key+'_only_fake'][0:1,1:2,:,:,h//2]), dim=0)
                            
                                recon_loss = 1 - avg_dsc(val_batch,source_key=label_key+'_pred', target_key=label_key+'_recon_pred',botindex=1,topindex=len(mask_index),return_mean=True)
                                dsc_loss = 1 - avg_dsc(val_batch,source_key=label_key+'_pred', target_key=label_key+'_only',botindex=1,topindex=len(mask_index),return_mean=True)
                                dsc_loss_fake = 1 - avg_dsc(val_batch,source_key=label_key+'_pred', target_key=label_key+'_only_fake',botindex=1,topindex=len(mask_index),return_mean=True)

                                if only_pseudo:
                                    final_loss = dsc_loss_fake
                                elif domain_loss_type == 8:
                                    if recon_loss < 0.15: cur_lambda = lambda_vae * 0.6
                                    elif recon_loss < 0.225: cur_lambda = lambda_vae * 1.2
                                    elif recon_loss < 0.3: cur_lambda = lambda_vae * 2.0
                                    else: cur_lambda = lambda_vae * 3.0
                                    if cur_lambda > 1:
                                        if kl: final_loss = recon_loss + klloss + 1 / cur_lambda * dsc_loss_fake
                                        else: final_loss = recon_loss + 1 / cur_lambda * dsc_loss_fake
                                    else:
                                        if kl: final_loss = cur_lambda * (recon_loss + klloss) + dsc_loss_fake
                                        else: final_loss = cur_lambda * recon_loss + dsc_loss_fake
                                elif domain_loss_type == 9:
                                    if recon_loss < 0.15: cur_lambda = lambda_vae * 0.6
                                    elif recon_loss < 0.225: cur_lambda = lambda_vae * 1.2
                                    elif recon_loss < 0.3: cur_lambda = lambda_vae * 2.0
                                    else: cur_lambda = lambda_vae * 3.0
                                    final_loss = (cur_lambda * recon_loss + dsc_loss_fake) / (1 + cur_lambda)
                                elif domain_loss_type == 10:
                                    loss_square = torch.mean(torch.square(val_batch[label_key+'_pred'])) 
                                    final_loss = loss_square + recon_loss + dsc_loss_fake
                                    print(final_loss)
                                elif domain_loss_type == 11:
                                    final_loss = lambda_vae * recon_loss + dsc_loss_fake + recon_loss * dsc_loss_fake
                                elif domain_loss_type == 12:
                                    final_loss = lambda_vae * recon_loss + dsc_loss_fake + (1 - recon_loss) * (1 - dsc_loss_fake)
                                elif domain_loss_type == 13:
                                    recon_loss -= 0.15
                                    recon_loss = torch.maximum(recon_loss, 0)
                                    final_loss = lambda_vae * recon_loss
                                elif domain_loss_type == 14:
                                    recon_loss -= 0.1
                                    recon_loss = torch.maximum(recon_loss, 0)
                                    final_loss = lambda_vae * recon_loss + dsc_loss_fake
                                elif domain_loss_type == 15:
                                    recon_loss -= 0.1
                                    recon_loss[recon_loss < 0] = 0
                                    dsc_loss_fake -= 0.1
                                    dsc_loss_fake[dsc_loss_fake < 0] = 0
                                    final_loss = lambda_vae * recon_loss + dsc_loss_fake
                                elif turn_epoch != -1:
                                    if (epoch // turn_epoch) % 2 == 0:
                                        final_loss = lambda_vae * recon_loss
                                    else:
                                        final_loss = lambda_vae * recon_loss + dsc_loss_fake
                                elif epoch >= lambda_vae_warmup:
                                    final_loss = lambda_vae * recon_loss + dsc_loss_fake
                                else:
                                    final_loss = lambda_vae * epoch / lambda_vae_warmup * recon_loss + dsc_loss_fake

                                optimizer_finetune = torch.optim.SGD(model_finetune.parameters(),
                                    lr=lr_finetune,weight_decay = weight_decay,momentum=0)
                                # if recon_loss < 0.15: break
                                optimizer_finetune.zero_grad()
                                final_loss.backward()
                                optimizer_finetune.step()
                                loss = []
                                display_image={}
                                loss.append(['finetune_recon_loss',recon_loss.item()])
                                loss.append(['finetune_dice_loss_fake',dsc_loss_fake.item()])
                                loss.append(['finetune_dice_loss',dsc_loss.item()])
                                loss.append(['finetune_final_loss',final_loss.item()])
                                loss.append(['finetune_epoch',epoch])
                                display_image.update({label_key+'_display_finetune':val_batch[label_key+'_display']})
                                saver.write_display(i+val_idx*val_finetune+epoch*(max_idx_in_epoch+1)*2,loss,display_image,force_write=True)

                    with torch.no_grad():
                        val_batch[label_key+'_only'] = val_batch[label_key].type(torch.cuda.LongTensor)
                        one_hot = torch.cuda.FloatTensor(val_batch[label_key+'_only'].size(0),len(mask_index),val_batch[label_key+'_only'].size(2),val_batch[label_key+'_only'].size(3),val_batch[label_key+'_only'].size(4)).zero_()
                        val_batch[label_key+'_only'] = one_hot.scatter_(1,val_batch[label_key+'_only'].data,1)
                        val_batch[img_key] = val_batch[img_key].cuda()
                        if method == 'domain_adaptation':
                            if val_finetune != 0:
                                val_batch = model(val_batch,img_key,label_key+'_pred_noft',label_key+'_recon_pred_noft')
                                val_batch = model_finetune(val_batch,img_key,label_key+'_pred',label_key+'_recon_pred')
                                val_batch[label_key+'_only_recon'],_,_ = model_finetune.Vae(val_batch[label_key+'_only'],if_random=False,scale=0)
                            else:
                                val_batch = model(val_batch,img_key,label_key+'_pred',label_key+'_recon_pred')
                                val_batch[label_key+'_only_recon'],_,_ = model.Vae(val_batch[label_key+'_only'],if_random=False,scale=0)
                            if save_more_reference and val_idx == epoch % len(val_loader):
                                h=val_batch[label_key+'_pred'][0:1,0:1,:,:,:].shape[4]
                                val_batch[label_key+'_display']= torch.cat((val_batch[label_key+'_recon_pred'][0:1,1:2,:,:,h//2], \
                                    val_batch[label_key+'_only'][0:1,1:2,:,:,h//2],val_batch[label_key+'_pred'][0:1,1:2,:,:,h//2]), dim=0)
                                display_image.update({label_key+'_display_val':val_batch[label_key+'_display']})
                            if analysis_figure_name != None:
                                val_batch = model_fix_parallel(val_batch,img_key,label_key+'_only_fake',label_key+'_only_fake_recon')
                            if save_eval_result and epoch % 10 == 0:
                                filename = os.path.join(result_path, f'{epoch}_{val_idx}_pred.join')
                                # torch.save(binarize(val_batch[label_key+'_pred']), filename)
                                np.save(filename, binarize(val_batch[label_key+'_pred']).cpu().detach().numpy())
                                filename = os.path.join(result_path, f'{epoch}_{val_idx}_pic')
                                np.save(filename, val_batch[img_key].cpu().detach().numpy())
                                # torch.save(val_batch[img_key], filename)
                                filename = os.path.join(result_path, f'{epoch}_{val_idx}_gt')
                                np.save(filename, binarize(val_batch[label_key+'_only']).cpu().detach().numpy())
                                # filename = os.path.join(result_path, f'{epoch}_{val_idx}_recon')
                                # np.save(filename, binarize(val_batch[label_key+'_recon_pred']).cpu().detach().numpy())
                                # filename = os.path.join(result_path, f'{epoch}_{val_idx}_pseu')
                                # np.save(filename, binarize(val_batch[label_key+'_only_fake']).cpu().detach().numpy())
                                # filename = os.path.join(result_path, f'{epoch}_{val_idx}_gt_recon')
                                # np.save(filename, binarize(val_batch[label_key+'_only_recon']).cpu().detach().numpy())
                                # torch.save(val_batch[label_key+'_only'], filename)
                        elif method == 'domain_adaptation_dis':
                            val_batch = model(val_batch,img_key,label_key+'_pred',label_key+'_score')
                            if save_eval_result and epoch % 10 == 0:
                                filename = os.path.join(result_path, f'{epoch}_{val_idx}_pred.join')
                                np.save(filename, binarize(val_batch[label_key+'_pred']).cpu().detach().numpy())
                                filename = os.path.join(result_path, f'{epoch}_{val_idx}_pic')
                                np.save(filename, val_batch[img_key].cpu().detach().numpy())
                                filename = os.path.join(result_path, f'{epoch}_{val_idx}_gt')
                                np.save(filename, binarize(val_batch[label_key+'_only']).cpu().detach().numpy())
                        else:
                            assert False
                        
                        if val_finetune != 0:
                            score_noft[val_idx] = avg_dsc(val_batch,source_key=label_key+'_pred_noft', target_key=label_key+'_only', binary=True,botindex=1,topindex=len(mask_index)).item()
                            dsc_pancreas_noft += score_noft[val_idx]

                        score[val_idx] = avg_dsc(val_batch,source_key=label_key+'_pred', target_key=label_key+'_only', binary=True,botindex=1,topindex=len(mask_index)).item()
                        dsc_pancreas += score[val_idx]
                        if analysis_figure_name != None:
                            gt_recon_loss = 1 - avg_dsc(val_batch,source_key=label_key+'_only_recon', target_key=label_key+'_only', binary=True,botindex=1,topindex=len(mask_index)).item()
                            gt_dsc_loss_fake = 1 - avg_dsc(val_batch,source_key=label_key+'_only_fake', target_key=label_key+'_only',botindex=1,topindex=len(mask_index),return_mean=True).item()
                            recon_loss = 1 - avg_dsc(val_batch,source_key=label_key+'_pred', target_key=label_key+'_recon_pred', binary=True,botindex=1,topindex=len(mask_index)).item()
                            dsc_loss_fake = 1 - avg_dsc(val_batch,source_key=label_key+'_pred', target_key=label_key+'_only_fake',botindex=1,topindex=len(mask_index),return_mean=True).item()
                            pseudo_recon_loss = 1 - avg_dsc(val_batch,source_key=label_key+'_only_fake', target_key=label_key+'_only_fake_recon',botindex=1,topindex=len(mask_index),return_mean=True).item()
                            pseudo_dsc_loss_fake = 1 - avg_dsc(val_batch,source_key=label_key+'_only_fake', target_key=label_key+'_only_fake',botindex=1,topindex=len(mask_index),return_mean=True).item()
                            print(f"result: {val_idx}")
                            print(1 - score[val_idx])
                            print(gt_recon_loss)
                            print(recon_loss)
                            print(pseudo_recon_loss)
                            print(dsc_loss_fake)
                            print(gt_dsc_loss_fake)
                            print(pseudo_dsc_loss_fake)
                            loss_gt += gt_recon_loss
                            loss_recon += recon_loss
                            loss_fake += dsc_loss_fake
                            score_figure[val_idx] = [dsc_loss_fake, recon_loss]
                            score_figure_gt[val_idx] = [gt_dsc_loss_fake, gt_recon_loss]
                            score_figure_pseudo[val_idx] = [pseudo_dsc_loss_fake, pseudo_recon_loss]
                    

                dsc_pancreas /= (val_idx+1)
                if val_finetune != 0:
                    dsc_pancreas_noft /= (val_idx+1)
                if analysis_figure_name != None:
                    loss_gt /= (val_idx + 1)
                    loss_recon /= (val_idx + 1)
                    loss_fake /= (val_idx + 1)
                    print("gt_recon_loss")
                    print(loss_gt)
                    print("recon_loss")
                    print(loss_recon)
                    print("fake_loss")
                    print(loss_fake)
                    scatter_plot(score_figure, analysis_figure_name, "Pseudo_loss", "Recon_loss")
                    scatter_plot(score_figure_gt, analysis_figure_name + '_gt', "Pseudo_loss", "Recon_loss")
                    scatter_plot(score_figure_pseudo, analysis_figure_name + '_pseudo', "Pseudo_loss", "Recon_loss")
                    scatter_plot_multi(score_figure, score_figure_gt, 'analysis')


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
                            if method == 'domain_adaptation':
                                tr_batch = model_fix_parallel(tr_batch,img_key,label_key+'_only_fake',label_key+'_asdfasdf')
                                tr_batch[label_key+'_only_fake'] = binarize(tr_batch[label_key+'_only_fake'])
                                tr_batch[label_key+'_display']= torch.cat((tr_batch[label_key+'_recon_pred'][0:1,1:2,:,:,h//2], \
                                    tr_batch[label_key+'_only'][0:1,1:2,:,:,h//2],tr_batch[label_key+'_pred'][0:1,1:2,:,:,h//2], \
                                    tr_batch[label_key+'_only_fake'][0:1,1:2,:,:,h//2]), dim=0)
                            else:
                                tr_batch[label_key+'_display']= torch.cat((tr_batch[label_key+'_recon_pred'][0:1,1:2,:,:,h//2], \
                                    tr_batch[label_key+'_only'][0:1,1:2,:,:,h//2],tr_batch[label_key+'_pred'][0:1,1:2,:,:,h//2]), dim=0)
                        else:
                            assert False

                        display_image.update({label_key+'_display_train':tr_batch[label_key+'_display']})
            
            output_score = os.path.join(display_path, f"score_{epoch}.json")
            with open(output_score, "w") as f:
                json.dump(score, f)
            if val_finetune != 0:
                output_score = os.path.join(display_path, f"score_noft_{epoch}.json")
                with open(output_score, "w") as f:
                    json.dump(score_noft, f)

            loss = []
            loss.append(['val_result', dsc_pancreas])
            if val_finetune != 0: loss.append(['val_result_no_finetune', dsc_pancreas_noft])
            saver.write_display((epoch+1)*(max_idx_in_epoch+1), loss, display_image, force_write=True)
            print('epoch %d validation result: %f, best result %f.' % (epoch+1, dsc_pancreas, best_result))
            time2 = time.time()
            interval = time2 - time1
            print('Time: {}'.format(interval))
            
            if test_only: break

            model.train()
            if method=='joint_train' or method=='sep_joint_train' or method=='domain_adaptation':
                model.Vae.eval()
            if method == 'domain_adaptation_dis':
                model.Dis.eval()

            if dsc_pancreas > best_result:
                best_result = dsc_pancreas
                torch.save({
                        'epoch': (epoch+1)*eval_epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                        }, os.path.join(save_path,'best_model.ckpt'))
        
        ## save model
        if (epoch+1) % (save_epoch // eval_epoch) == 0:
            print('saving model')
            torch.save({
                        'epoch': (epoch+1)*eval_epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                        }, os.path.join(save_path,'model_epoch'+str((epoch+1)*eval_epoch)+'.ckpt'))

    print('Finished Training')