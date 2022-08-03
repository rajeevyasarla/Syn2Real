import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import pdb
import numpy as np
import random
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import euclidean_distances
from numpy.linalg import inv
from sklearn.metrics.pairwise import cosine_similarity


def kronecker(A, B):
    AB = torch.einsum("ab,cd->acbd", A, B)
    AB = AB.view(A.size(0)*B.size(0), A.size(1)*B.size(1))
    return AB
# numpy version of Linear kernel function 
def kernel_linear(X_u,X_l):
    xu_shape = X_u.shape
    xl_shape = X_l.shape
    # pdb.set_trace()
    if len(xu_shape)==2 :
        X_u_norm = X_u / X_u.norm(dim=-1).view(xu_shape[0],1)
        X_l_norm = X_l / X_l.norm(dim=-1).view(xl_shape[0],1)
        ker_t = torch.mm(X_u_norm, X_l_norm.transpose(0,1))
    elif len(xu_shape)==3 :
        X_u_norm = X_u / X_u.norm(dim=-1).view(xu_shape[0],xu_shape[1],1)
        X_l_norm = X_l / X_l.norm(dim=-1).view(xl_shape[0],xl_shape[1],1)
        ker_t = torch.bmm(X_u_norm, X_l_norm.transpose(1,2))
    elif len(xu_shape)==4 :
        X_u = X_u.reshape(xu_shape[0]*xu_shape[1],xu_shape[2],xu_shape[3])
        X_l = X_l.reshape(xl_shape[0]*xl_shape[1],xl_shape[2],xl_shape[3])
        X_u_norm = X_u / X_u.norm(dim=-1).view(xu_shape[0]*xu_shape[1],xu_shape[2],1)
        X_l_norm = X_l / X_l.norm(dim=-1).view(xl_shape[0]*xl_shape[1],xl_shape[2],1)
        X_l_norm = X_l_norm.transpose(1,2)        
        ker_t = torch.bmm(X_u_norm, X_l_norm)
        ker_t = ker_t.view(xu_shape[0],xu_shape[1],xu_shape[2],xl_shape[2])
    return ker_t
pdist_ker = nn.PairwiseDistance(p=2)
def kernel_distance(X_u,X_l):
    xu_shape = X_u.shape
    xl_shape = X_l.shape
    # pdb.set_trace()
    if len(xu_shape)==2 :
        x_l_t = X_l.repeat(xu_shape[0],1)
        x_u_t = X_u.repeat(1,xl_shape[0]).view(xl_shape[0]*xu_shape[0],xu_shape[1])
        ker_t = pdist_ker(x_u_t,x_l_t)
        ker_t = ker_t.view(xu_shape[0],xl_shape[0])
    elif len(xu_shape)==3 :
        x_l_t = X_l.repeat(1,xu_shape[1],1)
        x_u_t = X_u.repeat(1,1,xl_shape[1]).view(xu_shape[0],xl_shape[1]*xu_shape[1],xu_shape[2])
        ker_t = pdist_ker(x_u_t,x_l_t)
        ker_t = ker_t.view(xu_shape[0],xu_shape[1],xl_shape[1])
    elif len(xu_shape)==4 :
        X_u = X_u.reshape(xu_shape[0]*xu_shape[1],xu_shape[2],xu_shape[3])
        X_l = X_l.reshape(xl_shape[0]*xl_shape[1],xl_shape[2],xl_shape[3])
        x_l_t = X_l.repeat(1,xu_shape[2],1)
        x_u_t = X_u.repeat(1,1,xl_shape[2]).view(xu_shape[0]*xu_shape[1],xl_shape[2]*xu_shape[2],xu_shape[3])
        ker_t = pdist_ker(x_u_t,x_l_t)
        ker_t = ker_t.view(xu_shape[0],xu_shape[1],xu_shape[2],xl_shape[2])
    return ker_t

def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    distt  =torch.clamp(dist, 0.0, np.inf)
    dist[dist != dist] = 0
    return dist

# torch version of Squared Exponential kernel function 
def kernel_se(x,y,var):
    sigma_1 = 1.0
    pw = 0.6
    l_1 = torch.max(var)#.max(axis=-1).max(axis=-1).max(axis=-1)#1.0#(np.sum(mu**2))**(pw)
    d = kernel_distance(x,y)
    Ker = sigma_1**2 *torch.exp(-0.5*d/l_1**2)
    return Ker

# numpy version of Squared Exponential kernel function 
def kernel_se_np(x,y,var):
    sigma_1 = 1.0
    pw = 0.6
    l_1 = var.max(axis=-1).max(axis=-1).max(axis=-1)#1.0#(np.sum(mu**2))**(pw)
    d = cdist(x,y)**2
    Ker = sigma_1**2 * np.exp(-0.5*d/l_1**2)
    return Ker

# torch version of Rational Quadratic kernel function
def kernel_rq(x,y,var,alpha=0.5):
    sigma_1 = 1.0
    pw = 0.6
    l_1 = torch.max(var)#var.max(axis=-1).max(axis=-1).max(axis=-1)#1.0#(np.sum(mu**2))**(pw)
    d = kernel_distance(x,y)
    Ker = sigma_1**2 *(1+(0.5*d/(alpha*l_1**2)))**(-1*alpha)
    return Ker

# numpy version of Rational Quadratic kernel function 
def kernel_rq_np(x,y,var,alpha=0.5):
    sigma_1 = 1.0
    pw = 0.6
    l_1 = var.max(axis=-1).max(axis=-1).max(axis=-1)#1.0#(np.sum(mu**2))**(pw)
    d = cdist(x,y)**2
    Ker = sigma_1**2 * (1+(0.5*d/(alpha*l_1**2)))**(-1*alpha)
    return Ker



class GPStruct(object):
    def __init__(self,num_lbl,num_unlbl,train_batch_size,version,kernel_type):
        self.num_lbl = num_lbl # number of labeled images
        self.num_unlbl = num_unlbl # number of unlabeled images
        self.z_height=32 # height of the feature map z i.e dim 2 
        self.z_width = 32 # width of the feature map z i.e dim 3
        self.z_numchnls = 32 # number of feature maps in z i.e dim 1
        self.num_nearest = 16 #number of nearest neighbors for unlabeled vector
        self.Fz_lbl = torch.zeros((self.num_lbl,self.z_numchnls,self.z_height,self.z_width),dtype=torch.float32).cuda() #Feature matrix Fzl for latent space labeled vector matrix
        self.Fz_unlbl = torch.zeros((self.num_unlbl,self.z_numchnls,self.z_height,self.z_width),dtype=torch.float32).cuda() #Feature matrix Fzl for latent space unlabeled vector matrix
        self.ker_lbl = torch.zeros((self.num_lbl,self.num_lbl)).cuda() # kernel matrix of labeled vectors
        self.ker_lbl_ang = torch.zeros((self.num_lbl,self.num_lbl)).cuda() # kernel matrix of labeled vectors
        self.ker_unlbl = torch.zeros((self.num_unlbl,self.num_lbl)).cuda() # kernel matrix of unlabeled vectors
        self.ker_unlbl_ang = torch.zeros((self.num_unlbl,self.num_lbl)).cuda() # kernel matrix of unlabeled vectors
        # self.metric_l = nn.Parameter(torch.rand((32,32),dtype=torch.float32), requires_grad=True)
        # self.metric_l = self.metric_l.cuda()
        self.sigma_noise = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        # self.metric_m = torch.matmul(self.metric_l.t(),self.metric_l)
        self.dict_lbl ={} # dictionary helpful in saving the feature vectors
        self.dict_unlbl ={} # dictionary helpful in saving the feature vectors
        self.lambda_var = 0.33 # factor multiplied with minimizing variance
        self.train_batch_size = train_batch_size
        self.version = version # version1 is GP SIMO model and version2 is GP MIMO model
        self.kernel_type = kernel_type
        self.KL_div = torch.nn.KLDivLoss()
        # declaring kernel function
        if kernel_type =='Linear':
            self.kernel_comp = kernel_linear
        elif kernel_type =='Squared_exponential':
            self.kernel_comp = kernel_se
        elif kernel_type =='Rational_quadratic':
            self.kernel_comp = kernel_rq

        if kernel_type =='Linear':
            self.kernel_comp_np = cosine_similarity
        elif kernel_type =='Squared_exponential':
            self.kernel_comp_np = kernel_se_np
        elif kernel_type =='Rational_quadratic':
            self.kernel_comp_np = kernel_rq_np

    

    def gen_featmaps_unlbl(self,dataloader,net,device):
        print("Unlabelled: started storing feature vectors and kernel matrix")
        count =0
        for batch_id, train_data in enumerate(dataloader):

            input_im, gt, imgid = train_data
            input_im = input_im.to(device)
            gt = gt.to(device)

            
            net.eval()
            pred_image,zy_in = net(input_im)
            tensor_mat = zy_in.data#torch.squeeze(zy_in.data)
            # saving latent space feature vectors 
            for i in range(tensor_mat.shape[0]):
                if imgid[i] not in self.dict_unlbl.keys():
                    self.dict_unlbl[imgid[i]] = count
                    count += 1
                tmp_i = self.dict_unlbl[imgid[i]]
                self.Fz_unlbl[tmp_i,:,:,:] = tensor_mat[i,:,:,:].data
                # tensor = torch.squeeze(tensor_mat[i,:,:,:])
        X = self.Fz_unlbl.view(-1,self.z_numchnls*self.z_height*self.z_width)
        Y = self.Fz_lbl.view(-1,self.z_numchnls*self.z_height*self.z_width)
        dist = torch.from_numpy(euclidean_distances(X.cpu().numpy(),Y.cpu().numpy())).cuda()
        self.ker_unlbl = torch.exp(-0.5*dist)
        self.ker_unlbl_ang = kernel_linear(X,Y)

        print("Unlabelled: stored feature vectors and kernel matrix")
        return

    def gen_featmaps(self,dataloader,net,device):
        
        count =0
        print("Labelled: started storing feature vectors and kernel matrix")
        for batch_id, train_data in enumerate(dataloader):

            input_im, gt, imgid = train_data
            input_im = input_im.to(device)
            gt = gt.to(device)
            
            net.eval()
            pred_image,zy_in = net(input_im)
            tensor_mat = zy_in.data#torch.squeeze(zy_in.data)
            # saving latent space feature vectors
            for i in range(tensor_mat.shape[0]):
                if imgid[i] not in self.dict_lbl.keys():
                    self.dict_lbl[imgid[i]] = count
                    count += 1
                tmp_i = self.dict_lbl[imgid[i]]
                self.Fz_lbl[tmp_i,:,:,:] = tensor_mat[i,:,:,:].data
                # tensor = torch.squeeze(tensor_mat[i,:,:,:])
        X = self.Fz_lbl.view(-1,self.z_numchnls*self.z_height*self.z_width)
        Y = self.Fz_lbl.view(-1,self.z_numchnls*self.z_height*self.z_width)
        self.var_Fz_lbl = torch.std(self.Fz_lbl,axis=0)
        self.ker_lbl_ang = kernel_linear(X,Y)
        
        # dist = euclidean_distances(X,Y)**2
        dist = torch.from_numpy(euclidean_distances(X.cpu().numpy(),Y.cpu().numpy())).cuda()
        self.ker_lbl = torch.exp(-0.5*dist**2)

        print("Labelled: stored feature vectors and kernel matrix")
        return
    def loss(self,pred,target):
        # pred = pred.view(-1,self.z_height*self.z_width)
        # target = target.view(-1,self.z_height*self.z_width)
        diff = pred - target
        loss = diff**2#torch.matmul(self.metric_m,diff))
        return loss.mean(dim=-1).mean(dim=-1)
    def compute_gploss(self,zy_in,imgid,batch_id,reject_trsh=False,label_flg=0):
        tensor_mat = zy_in
        
        Sg_Pred = torch.zeros([self.train_batch_size,1])
        Sg_Pred = Sg_Pred.cuda()
        LSg_Pred = torch.zeros([self.train_batch_size,1])
        LSg_Pred = LSg_Pred.cuda()
        gp_loss = 0

        B,N,H,W = tensor_mat.shape
        tmp_i = [self.dict_unlbl[i] for i in imgid]
        tensor_vec = tensor_mat.view(-1,self.z_numchnls,1,self.z_height*self.z_width) 
        multiplier = torch.ones((B,1)).cuda()
        kernel_values =  self.ker_unlbl[tmp_i,:].data if label_flg==0 else self.ker_lbl[tmp_i,:].data
        # pdb.set_trace()
        # print(multiplier,torch.max(kernel_values.topk(k = self.num_nearest//4,dim=-1)[0],dim=1)[0])
        if self.version == 'version1':
            tensor_vec = tensor_mat.view(-1,1,self.z_numchnls*self.z_height*self.z_width) # z tensor to a vector
            if self.kernel_type =='Linear':
                ker_UU = self.kernel_comp(tensor_vec,tensor_vec) # k(z,z), i.e kernel value for z,z
            else :
                ker_UU = self.kernel_comp(tensor_vec,tensor_vec,self.var_Fz_lbl)
        else :
            if self.kernel_type =='Linear':
                ker_UU = self.kernel_comp(tensor_vec,tensor_vec) # k(z,z), i.e kernel value for z,z
            else :
                ker_UU = self.kernel_comp(tensor_vec,tensor_vec,self.var_Fz_lbl)

        nearest_vl = self.ker_unlbl_ang[tmp_i,:] if label_flg==0 else self.ker_lbl_ang[tmp_i,:] #kernel values are used to get neighbors

        nn_val,nn_ind = kernel_values.topk(k = self.num_nearest,dim=-1)
        near_vec_lbl = self.Fz_lbl[nn_ind.view(-1),:,:,:]
        if self.version == 'version1':
            near_vec_lbl = near_vec_lbl.view(B,self.num_nearest,self.z_numchnls*self.z_height*self.z_width)
            if self.kernel_type =='Linear':
                ker_LL = self.kernel_comp(near_vec_lbl,near_vec_lbl)
            else :
                ker_LL = self.kernel_comp(near_vec_lbl,near_vec_lbl,self.var_Fz_lbl)

            Eye = torch.eye(self.num_nearest)
            Eye = Eye.view(1,self.num_nearest,self.num_nearest).cuda()
            Eye = Eye.repeat(B,1,1)

            ker_LL = ker_LL + (self.sigma_noise**2)*Eye
            inv_ker_LL = torch.linalg.inv(ker_LL)
            tensor_vec = tensor_vec.view(B,1,self.z_numchnls*self.z_height*self.z_width)
            if self.kernel_type =='Linear':
                ker_UL = self.kernel_comp(tensor_vec,near_vec_lbl) 
            else :
                ker_UL = self.kernel_comp(tensor_vec,near_vec_lbl,self.var_Fz_lbl) 

            mean_pred = torch.bmm(ker_UL,torch.bmm(inv_ker_LL,near_vec_lbl))
            Eye = torch.eye(1)
            Eye = Eye.view(1,1,1).cuda()
            Eye = Eye.repeat(B,1,1)
            sigma_est = ker_UU - torch.bmm(ker_UL,torch.bmm(inv_ker_LL,ker_UL.transpose(1,2))) + (self.sigma_noise**2)*Eye
        else :
            near_vec_lbl = near_vec_lbl.view(B,self.num_nearest,self.z_numchnls,self.z_height*self.z_width)
            near_vec_lbl = near_vec_lbl.transpose(1,2)
            
            if self.kernel_type =='Linear':
                ker_LL = self.kernel_comp(near_vec_lbl,near_vec_lbl)
            else :
                ker_LL = self.kernel_comp(near_vec_lbl,near_vec_lbl,self.var_Fz_lbl)

            Eye = torch.eye(self.num_nearest)
            Eye = Eye.view(1,1,self.num_nearest,self.num_nearest).cuda()
            Eye = Eye.repeat(B,N,1,1)

            ker_LL = ker_LL + (self.sigma_noise**2)*Eye
            inv_ker_LL = torch.linalg.inv(ker_LL)
            tensor_vec = tensor_vec.view(B,self.z_numchnls,1,self.z_height*self.z_width)
            if self.kernel_type =='Linear':
                ker_UL = self.kernel_comp(tensor_vec,near_vec_lbl) 
            else :
                ker_UL = self.kernel_comp(tensor_vec,near_vec_lbl,self.var_Fz_lbl) 
            
            
            Eye = torch.eye(1)
            Eye = Eye.view(1,1,1).cuda()
            Eye = Eye.repeat(B*N,1,1)
            # pdb.set_trace()
            
            ker_UL = ker_UL.reshape(B*N,1,self.num_nearest)
            ker_LL = ker_LL.reshape(B*N,self.num_nearest,self.num_nearest)
            ker_UU = ker_UU.reshape(B*N,1,1)
            inv_ker_LL = inv_ker_LL.reshape(B*N,self.num_nearest,self.num_nearest)
            near_vec_lbl = near_vec_lbl.reshape(B*N,self.num_nearest,self.z_height*self.z_width)
            mean_pred = torch.bmm(ker_UL,torch.bmm(inv_ker_LL,near_vec_lbl))
            sigma_est = ker_UU - torch.bmm(ker_UL,torch.bmm(inv_ker_LL,ker_UL.transpose(1,2))) + (self.sigma_noise**2)*Eye
            sigma_est = sigma_est.reshape(B,N,1)
            tensor_vec = tensor_mat.view(-1,self.z_numchnls,self.z_height*self.z_width)
            # pdb.set_trace()
            # inv_sigma = torch.linalg.inv(sigma_est)
        for i in range(tensor_mat.shape[0]):
            if self.version == 'version1':
                loss_unsup = torch.mean((self.loss(tensor_vec[i,:,:],mean_pred[i,:,:]))/sigma_est[i,:,:]) + 1.0*self.lambda_var*torch.log(torch.det(sigma_est[i,:,:]))
            else :
                loss_unsup = torch.mean((self.loss(tensor_vec[i,:,:],mean_pred[i,:,:]))/sigma_est[i,:,:]) + 1.0*self.lambda_var*torch.mean(torch.log(torch.abs(sigma_est[i,:,:])))
                # loss_unsup = torch.mean(torch.matmul((tensor_vec[i,:,:]-mean_pred[i,:,:]).t(),torch.matmul(inv_sigma[i,:,:],(tensor_vec[i,:,:]-mean_pred[i,:,:])))) + 1.0*self.lambda_var*torch.log(torch.det(sigma_est[i,:,:])) 
            if loss_unsup==loss_unsup:
                gp_loss += torch.mean(multiplier[i]*((1.0*loss_unsup/self.train_batch_size)))
            Sg_Pred[i,:] = torch.mean(torch.log(torch.abs(sigma_est[i,:,:])))
        
        if not (batch_id % 100) and loss_unsup==loss_unsup:
            print(Sg_Pred.max().item(),gp_loss.item()/self.train_batch_size,Sg_Pred.mean().item(),torch.sum(multiplier))
        
        

        return gp_loss
