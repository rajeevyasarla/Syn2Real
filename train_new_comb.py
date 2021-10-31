

# --- Imports --- #
import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train_data import TrainData
from val_data import ValData
from model import DeRain_v2
from GP_new_fast import GPStruct
from utils import to_metrics, print_log, validation, adjust_learning_rate
from torchvision.models import vgg16
from perceptual import LossNetwork
import os
import numpy as np
import random
plt.switch_backend('agg')


# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-learning_rate', help='Set the learning rate', default=2e-4, type=float)
parser.add_argument('-crop_size', help='Set the crop_size', default=[256, 256], nargs='+', type=int)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=18, type=int)
parser.add_argument('-epoch_start', help='Starting epoch number of the training', default=0, type=int)
parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-category', help='Set image category (derain or dehaze?)', default='derain', type=str)
parser.add_argument('-version', help='Set the GP model (version1 or version2?)', default='version1', type=str)
parser.add_argument('-kernel_type', help='Set the GP model (Linear or Squared_exponential or Rational_quadratic?)', default='Linear', type=str)
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', type=str)
parser.add_argument('-lambda_GP', help='Set the lambda_GP for gploss in loss function', default=0.0015, type=float)
parser.add_argument('-seed', help='set random seed', default=19, type=int)
args = parser.parse_args()

learning_rate = args.learning_rate
crop_size = args.crop_size
train_batch_size = args.train_batch_size
epoch_start = args.epoch_start
lambda_loss = args.lambda_loss
val_batch_size = args.val_batch_size
category = args.category
version = args.version # version1 is GP model used in conference paper and version2 is GP model (feature level GP) used in journal paper
kernel_type = args.kernel_type
exp_name = args.exp_name
lambgp = args.lambda_GP
use_GP_inlblphase = False # indication whether or not to use GP during labeled phase

#set seed
seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed) 
    print('Seed:\t{}'.format(seed))

print('--- Hyper-parameters for training ---')
print('learning_rate: {}\ncrop_size: {}\ntrain_batch_size: {}\nval_batch_size: {}\nlambda_loss: {}\ncategory: {}'.format(learning_rate, crop_size,
      train_batch_size, val_batch_size, lambda_loss, category))

# --- Set category-specific hyper-parameters  --- #
if category == 'derain':
    num_epochs = 200
    train_data_dir = './data/train/derain/'
    val_data_dir = './data/test/derain/'
elif category == 'dehaze':
    num_epochs = 10
    train_data_dir = './data/train/dehaze/'
    val_data_dir = './data/test/dehaze/'
else:
    raise Exception('Wrong image category. Set it to derain or dehaze dateset.')


# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- Define the network --- #
net = DeRain_v2()


# --- Build optimizer --- #
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


# --- Multi-GPU --- #
net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)


# --- Define the perceptual loss network --- #
vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.to(device)
# vgg_model = nn.DataParallel(vgg_model, device_ids=device_ids)
for param in vgg_model.parameters():
    param.requires_grad = False
loss_network = LossNetwork(vgg_model)
loss_network.eval()


# --- Load the network weight --- #
if os.path.exists('./{}/'.format(exp_name))==False:     
    os.mkdir('./{}/'.format(exp_name))  
try:
    net.load_state_dict(torch.load('./{}/{}_best'.format(exp_name,category)))
    print('--- weight loaded ---')
except:
    print('--- no weight loaded ---')


# --- Calculate all trainable parameters in network --- #
pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Total_params: {}".format(pytorch_total_params))


# --- Load training data and validation/test data --- #
labeled_name = 'DIDMDN.txt'
unlabeled_name = 'JORDER_200L.txt'
val_filename = 'JORDER_200L.txt'
# --- Load training data and validation/test data --- #
unlbl_train_data_loader = DataLoader(TrainData(crop_size, train_data_dir,unlabeled_name), batch_size=train_batch_size, shuffle=True, num_workers=8)
lbl_train_data_loader = DataLoader(TrainData(crop_size, train_data_dir,labeled_name), batch_size=train_batch_size, shuffle=True, num_workers=8)
val_data_loader = DataLoader(ValData(val_data_dir,val_filename), batch_size=val_batch_size, shuffle=False, num_workers=8)

num_labeled = train_batch_size*len(lbl_train_data_loader) # number of labeled images
num_unlabeled = train_batch_size*len(unlbl_train_data_loader) # number of unlabeled images






# --- Previous PSNR and SSIM in testing --- #
net.eval()
old_val_psnr, old_val_ssim = validation(net, val_data_loader, device, category, exp_name)
print('old_val_psnr: {0:.2f}, old_val_ssim: {1:.4f}'.format(old_val_psnr, old_val_ssim))
net.train()
#intializing GPStruct
gp_struct = GPStruct(num_labeled,num_unlabeled,train_batch_size,version,kernel_type)
gray = T.Grayscale(num_output_channels=3)
gray_aug = torch.nn.Sequential(
    T.Normalize((-0.5, -0.5, -0.5), (2, 2, 2)),
    gray,
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
)
label_train = 1
for epoch in range(epoch_start,num_epochs):
    psnr_list = []
    start_time = time.time()
    adjust_learning_rate(optimizer, epoch, category=category)
    print("----------------------- Labeled Phase -------------------------")
#-------------------------------------------------------------------------------------------------------------
    #Labeled phase
    if epoch > 0 :
        reject_trsh = True
    else :
        reject_trsh = False
    if lambgp!=0 and use_GP_inlblphase == True:
        gp_struct.gen_featmaps(lbl_train_data_loader,net,device)
    if lambgp!=0:
        gp_struct.gen_featmaps(lbl_train_data_loader,net,device)
        gp_struct.gen_featmaps_unlbl(unlbl_train_data_loader,net,device)
    if gp_struct.sigma_noise.data<0.1:
        gp_struct.sigma_noise.data = gp_struct.sigma_noise.data.clamp(0.1,100)
    for batch_id, (lbl_data,unlbl_data) in enumerate(zip(lbl_train_data_loader,unlbl_train_data_loader)):
        l_input_image, l_gt, l_imgid = lbl_data
        l_input_image = l_input_image.to(device)
        l_gt = l_gt.to(device)

        u_input_image, u_gt, u_imgid = unlbl_data
        u_input_image = u_input_image.to(device)

        # --- Zero the parameter gradients --- #
        optimizer.zero_grad()

        
        # --- Forward + Backward + Optimize --- #
        net.train()
        # ----- labeled loss -------------#
        l_pred_image,l_zy_in = net(l_input_image)

        smooth_loss = F.smooth_l1_loss(l_pred_image, l_gt)
        perceptual_loss = loss_network(l_pred_image, l_gt)
        l_gp_loss = 0
        if lambgp!=0 and use_GP_inlblphase == True:
            l_gp_loss = gp_struct.compute_gploss(l_zy_in, l_imgid,batch_id,reject_trsh,1)

        # ----- unlabeled loss -------------#
        u_pred_image,u_zy_in = net(u_input_image)
        u_gp_loss = 0
        
        if lambgp!=0:
            u_gp_loss = gp_struct.compute_gploss(u_zy_in, u_imgid,batch_id,reject_trsh,0)
        

        loss = 0
        if label_train !=0 :
            loss = smooth_loss + lambda_loss*perceptual_loss 
        loss += lambgp*l_gp_loss 
        # print(u_gp_loss,loss)
        loss += lambgp*u_gp_loss

        loss.backward()
        optimizer.step()

        # --- To calculate average PSNR --- #
        im_psnr,im_ssim = to_metrics(l_pred_image, l_gt)
        psnr_list.extend(im_psnr)

        if not (batch_id % 100):
            print('Epoch: {0}, Iteration: {1}'.format(epoch, batch_id))

    # --- Calculate the average training PSNR in one epoch --- #
    train_psnr = sum(psnr_list) / len(psnr_list)

    # --- Save the network parameters --- #
    torch.save(net.state_dict(), './{}/{}'.format(exp_name,category))

    # --- Use the evaluation model in testing --- #
    net.eval()

    val_psnr, val_ssim = validation(net, val_data_loader, device, category, exp_name)
    one_epoch_time = time.time() - start_time
    print_log(epoch+1, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, category, exp_name)

    # --- update the network weight --- #
    if val_psnr >= old_val_psnr:
        torch.save(net.state_dict(), './{}/{}_best'.format(exp_name,category))
        print('model saved')
        old_val_psnr = val_psnr
#-------------------------------------------------------------------------------------------------------------