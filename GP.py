import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


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
def kernel_linear(X_u,X_l,vector_length,Num_A,Num_B):
    
    x_l_t =  X_l.repeat(Num_A,1)
    x_u_t =  X_u.repeat(1,Num_B)
    x_u_t = x_u_t.view(Num_A*Num_B,vector_length)
    ker_t = torch.nn.functional.cosine_similarity(x_u_t,x_l_t)
    ker_t = ker_t.view(Num_A,Num_B)
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
    l_1 = var.max(axis=-1).max(axis=-1).max(axis=-1)#1.0#(np.sum(mu**2))**(pw)
    d = pairwise_distances(x,y)
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
    l_1 = var.max(axis=-1).max(axis=-1).max(axis=-1)#1.0#(np.sum(mu**2))**(pw)
    d = pairwise_distances(x,y)
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
        self.num_nearest = 8 #number of nearest neighbors for unlabeled vector
        self.Fz_lbl = np.zeros((self.num_lbl,self.z_numchnls,self.z_height,self.z_width),dtype=np.float32) #Feature matrix Fzl for latent space labeled vector matrix
        self.Fz_unlbl = np.zeros((self.num_unlbl,self.z_numchnls,self.z_height,self.z_width),dtype=np.float32) #Feature matrix Fzl for latent space unlabeled vector matrix
        self.ker_lbl = np.zeros((self.num_lbl,self.num_lbl)) # kernel matrix of labeled vectors
        self.ker_unlbl = np.zeros((self.num_unlbl,self.num_lbl)) # kernel matrix of unlabeled vectors
        self.dict_lbl ={} # dictionary helpful in saving the feature vectors
        self.dict_unlbl ={} # dictionary helpful in saving the feature vectors
        self.lambda_var = 0.33 # factor multiplied with minimizing variance
        self.train_batch_size = train_batch_size
        self.version = version # version1 is GP SIMO model and version2 is GP MIMO model
        self.kernel_type = kernel_type
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
                self.Fz_unlbl[tmp_i,:,:,:] = tensor_mat[i,:,:,:].cpu().numpy()
                tensor = torch.squeeze(tensor_mat[i,:,:,:])
        X = self.Fz_unlbl.reshape(-1,self.z_numchnls*self.z_height*self.z_width)
        Y = self.Fz_lbl.reshape(-1,self.z_numchnls*self.z_height*self.z_width)
        dist = euclidean_distances(X,Y)
        self.ker_unlbl = np.exp(-0.5*dist)
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
                self.Fz_lbl[tmp_i,:,:,:] = tensor_mat[i,:,:,:].cpu().numpy()
                tensor = torch.squeeze(tensor_mat[i,:,:,:])
        X = self.Fz_lbl.reshape(-1,self.z_numchnls*self.z_height*self.z_width)
        Y = self.Fz_lbl.reshape(-1,self.z_numchnls*self.z_height*self.z_width)
        self.var_Fz_lbl = np.std(self.Fz_lbl,axis=0)
        # dist = euclidean_distances(X,Y)**2
        dist = euclidean_distances(X,Y)
        self.ker_lbl = np.exp(-0.5*dist**2)
        print("Labelled: stored feature vectors and kernel matrix")
        return

    def compute_gploss(self,zy_in,imgid,batch_id,label_flg=0):
        tensor_mat = zy_in
        
        Sg_Pred = torch.zeros([self.train_batch_size,1])
        Sg_Pred = Sg_Pred.cuda()
        LSg_Pred = torch.zeros([self.train_batch_size,1])
        LSg_Pred = LSg_Pred.cuda()
        gp_loss = 0
        
        for i in range(tensor_mat.shape[0]):
            tmp_i = self.dict_unlbl[imgid[i]] if label_flg==0 else self.dict_lbl[imgid[i]] # imag_id in the dictionary
            tensor = tensor_mat[i,:,:,:] # z tensor 
            tensor_vec = tensor.view(-1,self.z_height*self.z_width) # z tensor to a vector
            
            if self.version == 'version1':
                tensor_vec = tensor.view(-1,self.z_numchnls*self.z_height*self.z_width) # z tensor to a vector
                if self.kernel_type =='Linear':
                    ker_UU = self.kernel_comp(tensor_vec,tensor_vec,self.z_height*self.z_width*self.z_numchnls,1,1) # k(z,z), i.e kernel value for z,z
                else :
                    ker_UU = self.kernel_comp(tensor_vec,tensor_vec,self.var_Fz_lbl)
            else :
                if self.kernel_type =='Linear':
                    ker_UU = self.kernel_comp(tensor_vec,tensor_vec,self.z_height*self.z_width,self.z_numchnls,self.z_numchnls) # k(z,z), i.e kernel value for z,z
                else :
                    ker_UU = self.kernel_comp(tensor_vec,tensor_vec,self.var_Fz_lbl)

            nearest_vl = self.ker_unlbl[tmp_i,:] if label_flg==0 else self.ker_lbl[tmp_i,:] #kernel values are used to get neighbors
            # Nearest and Farthest neighbors
            tp32_vec = np.array(sorted(range(len(nearest_vl)), key=lambda k: nearest_vl[k])[-1*self.num_nearest:])
            lt32_vec = np.array(sorted(range(len(nearest_vl)), key=lambda k: nearest_vl[k])[:self.num_nearest])

            # Nearest neighbor latent space labeled vectors
            near_dic_lbl = np.zeros((self.num_nearest,self.z_numchnls,self.z_height,self.z_width))
            for j in range(self.num_nearest):
                near_dic_lbl[j,:] = self.Fz_lbl[tp32_vec[j],:,:,:]
            if self.version == 'version1':
                near_vec_lbl = np.reshape(near_dic_lbl,(self.num_nearest,self.z_numchnls*self.z_height*self.z_width))
            else :
                near_vec_lbl = np.reshape(near_dic_lbl,(self.num_nearest*self.z_numchnls,self.z_height*self.z_width))
            # Farthest neighbor latent space labeled vectors
            far_dic_lbl = np.zeros((self.num_nearest,self.z_numchnls,self.z_height,self.z_width))
            for j in range(self.num_nearest):
                far_dic_lbl[j,:] = self.Fz_lbl[lt32_vec[j],:,:,:]
            if self.version == 'version1':
                far_vec_lbl = np.reshape(far_dic_lbl,(self.num_nearest,self.z_numchnls*self.z_height*self.z_width))
            else :
                far_vec_lbl = np.reshape(far_dic_lbl,(self.num_nearest*self.z_numchnls,self.z_height*self.z_width))

            # computing kernel matrix of labeled latent vectors 
            # and then computing (K_LL+sig^2I)^(-1)
            if self.kernel_type =='Linear':
                ker_LL = self.kernel_comp_np(near_vec_lbl,near_vec_lbl)
            else :
                ker_LL = self.kernel_comp_np(near_vec_lbl,near_vec_lbl,self.var_Fz_lbl)
            if self.version == 'version1':
                inv_ker = inv(ker_LL+1.0*np.eye(self.num_nearest))
            else :
                inv_ker = inv(ker_LL+1.0*np.eye(self.num_nearest*self.z_numchnls))
            if self.kernel_type =='Linear':
                farker_LL = self.kernel_comp_np(far_vec_lbl,far_vec_lbl)
            else :
                farker_LL = self.kernel_comp_np(far_vec_lbl,far_vec_lbl,self.var_Fz_lbl)
            if self.version == 'version1':
                farinv_ker = inv(farker_LL+1.0*np.eye(self.num_nearest))
            else :
                farinv_ker = inv(farker_LL+1.0*np.eye(self.num_nearest*self.z_numchnls))

            mn_pre = np.matmul(inv_ker,near_vec_lbl) # used for computing mean prediction
            mn_pre = mn_pre.astype(np.float32)

            #converting require variables to cuda tensors 
            near_vec_lbl = torch.from_numpy(near_vec_lbl.astype(np.float32))
            far_vec_lbl = torch.from_numpy(far_vec_lbl.astype(np.float32))
            inv_ker = torch.from_numpy(inv_ker.astype(np.float32))
            farinv_ker = torch.from_numpy(farinv_ker.astype(np.float32))
            inv_ker = inv_ker.cuda()
            farinv_ker = farinv_ker.cuda()
            near_vec_lbl = near_vec_lbl.cuda()
            far_vec_lbl = far_vec_lbl.cuda()

            mn_pre = torch.from_numpy(mn_pre) # used for mean prediction (mu) or z_pseudo 
            mn_pre = mn_pre.cuda()


            # Identity matrix
            Eye = torch.eye(self.z_numchnls)
            Eye = Eye.cuda()

            # computing sigma or variance between nearest labeled vectors and unlabeled vector
            if self.version == 'version1':
                if self.kernel_type =='Linear':
                    ker_UL = self.kernel_comp(tensor_vec,near_vec_lbl,self.z_height*self.z_width*self.z_numchnls,1,self.num_nearest)
                else :
                    ker_UL = self.kernel_comp(tensor_vec,near_vec_lbl,self.var_Fz_lbl)
                sigma_est = ker_UU - torch.matmul(ker_UL,torch.matmul(inv_ker,ker_UL.t())) + 1.0
                # computing variance between farthest labeled vectors and unlabeled vector
                if self.kernel_type =='Linear':
                    Farker_UL = self.kernel_comp(tensor_vec,far_vec_lbl,self.z_height*self.z_width*self.z_numchnls,1,self.num_nearest)
                else :
                    Farker_UL = self.kernel_comp(tensor_vec,far_vec_lbl,self.var_Fz_lbl)
                far_sigma_est = ker_UU - torch.matmul(Farker_UL,torch.matmul(farinv_ker,Farker_UL.t())) + 1.0
            else :
                # computing sigma or variance between nearest labeled vectors and unlabeled vector
                if self.kernel_type =='Linear':
                    ker_UL = self.kernel_comp(tensor_vec,near_vec_lbl,self.z_height*self.z_width,self.z_numchnls,self.z_numchnls*self.num_nearest)
                else :
                    ker_UL = self.kernel_comp(tensor_vec,near_vec_lbl,self.var_Fz_lbl)
                sigma_est = ker_UU - torch.matmul(ker_UL,torch.matmul(inv_ker,ker_UL.t())) + Eye
                # computing variance between farthest labeled vectors and unlabeled vector
                if self.kernel_type =='Linear':
                    Farker_UL = self.kernel_comp(tensor_vec,far_vec_lbl,self.z_height*self.z_width,self.z_numchnls,self.z_numchnls*self.num_nearest)
                else :
                    Farker_UL = self.kernel_comp(tensor_vec,far_vec_lbl,self.var_Fz_lbl)
                far_sigma_est = ker_UU - torch.matmul(Farker_UL,torch.matmul(farinv_ker,Farker_UL.t())) + Eye
            
            # computing mean prediction
            mean_pred = torch.matmul(ker_UL,mn_pre) #mean prediction (mu) or z_pseudo
            
            inv_sigma = torch.inverse(sigma_est)
            if self.version == 'version1':
                loss_unsup = torch.mean(((tensor_vec-mean_pred)**2)/sigma_est[0]) + 1.0*self.lambda_var*torch.log(torch.det(sigma_est)) - 0.000001*self.lambda_var*torch.log(torch.det(far_sigma_est))
            else:
                loss_unsup = torch.mean(torch.matmul((tensor_vec-mean_pred).t(),torch.matmul(inv_sigma,(tensor_vec-mean_pred)))) + 1.0*self.lambda_var*torch.log(torch.det(sigma_est))  #torch.mean(torch.matmul((tensor_vec-mean_pred).t(),torch.matmul(inv_sigma,(tensor_vec-mean_pred))))
            if loss_unsup==loss_unsup:
                gp_loss += ((1.0*loss_unsup/self.train_batch_size))
            Sg_Pred[i,:] = torch.log(torch.det(sigma_est))
            LSg_Pred[i,:] = torch.log(torch.det(far_sigma_est))
        
        if not (batch_id % 100) and loss_unsup==loss_unsup:
            print(LSg_Pred.max().item(),Sg_Pred.max().item(),gp_loss.item()/self.train_batch_size,Sg_Pred.mean().item())
        
        

        return gp_loss
