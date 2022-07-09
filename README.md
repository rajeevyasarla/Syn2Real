# Syn2Real 
Syn2Real Transfer Learning for Image Deraining using Gaussian Processes

[Rajeev Yasarla*](https://sites.google.com/view/rajeevyasarla/home), [Vishwanath A. Sindagi*](https://www.vishwanathsindagi.com/), [Vishal M. Patel](https://engineering.jhu.edu/ece/faculty/vishal-m-patel/)

[Paper Link](http://openaccess.thecvf.com/content_CVPR_2020/papers/Yasarla_Syn2Real_Transfer_Learning_for_Image_Deraining_Using_Gaussian_Processes_CVPR_2020_paper.pdf)(CVPR '20)

[Oral video Link](https://www.youtube.com/watch?v=iYuv4Cqgq4k)

    @InProceedings{Yasarla_2020_CVPR,
    author = {Yasarla, Rajeev and Sindagi, Vishwanath A. and Patel, Vishal M.},
    title = {Syn2Real Transfer Learning for Image Deraining Using Gaussian Processes},
    booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
    }
We propose a Gaussian Process-based semi-supervised learning framework which enables the network in learning to derain using synthetic dataset while generalizing better using  unlabeled real-world images. Through extensive experiments and ablations on several challenging datasets (such as Rain800, Rain200H and DDN-SIRR), we show that the proposed method, when trained on limited labeled data, achieves on-par performance with fully-labeled training. Additionally, we demonstrate that using unlabeled real-world images in the proposed GP-based framework results in superior performance as compared to existing methods.

## Journal extension:
Semi-Supervised Image Deraining using Gaussian Processes

[Paper Link](https://arxiv.org/pdf/2009.13075.pdf)

## Prerequisites:
1. Linux
2. Python 2 or 3
3. Pytorch version >=1.9
4. CPU or NVIDIA GPU + CUDA CuDNN (CUDA 8.0)

## Dataset structure
1. download the rain datasets and arrange the rainy images and clean images in the following order
2. Save the image names into text file (dataset_filename.txt)
```
   .
    ├── data 
    |   ├── train # Training  
    |   |   ├── derain        
    |   |   |   ├── <dataset_name>   
    |   |   |   |   ├── rain              # rain images 
    |   |   |   |   └── norain            # clean images
    |   |   |   └── dataset_filename.txt
    |   └── test  # Testing
    |   |   ├── derain         
    |   |   |   ├── <dataset_name>          
    |   |   |   |   ├── rain              # rain images 
    |   |   |   |   └── norain            # clean images
    |   |   |   └── dataset_filename.txt
```

## To test Syn2Real:
1. mention test dataset text file in the line 57 of test.py, for example
```    
    val_filename = 'SIRR_test.txt'
```
2. Run the following command
```   
    python test.py -category derain -exp_name DDN_SIRR_withGP
```
## To train Syn2Real:
1. mention the labeled, unlabeled, and validation dataset in lines 119-121 of train.py, for example
```
    labeled_name = 'DDN_100_split1.txt'
    unlabeled_name = 'real_input_split1.txt'
    val_filename = 'SIRR_test.txt'
``` 
2. Run the following command to train the base network without Gaussian processes
```
    python train.py  -train_batch_size 2  -category derain -exp_name DDN_SIRR_withoutGP  -lambda_GP 0.00 -epoch_start 0
```
3. Run the following command to train Syn2Real (CVPR'20) model 
```
    python train.py  -train_batch_size 2  -category derain -exp_name DDN_SIRR_withGP  -lambda_GP 0.0015 -epoch_start 0 -version version1
```
4. Run the following command to train Syn2Real++ (journal submission GP modellig at feature map level) 
```
    python train.py  -train_batch_size 2  -category derain -exp_name DDN_SIRR_withGP  -lambda_GP 0.0015 -epoch_start 0 -version version2
    
```
## Cross-domain experiments and Gaussian kernels
cross domain experiments are performed using DIDMDN dataset as source dataset, and other datasets like Rain800, JORDER_200L, DDN as target datasets.
```
----------------------------------------------------
Source datasets   | Target datasets                | 
----------------------------------------------------
DIDMDN            | Rain800, JORDER_200L, DDN      |
----------------------------------------------------
```
Gaussian processes can be modelled using different kernels like Linear or Squared_exponential or Rational_quadratic. the updated code provides an option to choose the kernel type
```
-kernel_type <Linear or Squared_exponential or Rational_quadratic>
```
## Fast version of GP

use GP_new_fast.py file for faster version of GP.
```
To use this GP_new_fast.py :
    comment line 14 in train.py
    and uncomment line 15 in train.py
```
Additionally you can use "train_new_comb.py" instead of "train.py". 

In "train_new_comb.py" does iterative training of the network, i.e. each iteration contains one labeled train step and one unlabeled train step.

Run the following command to train Syn2Real (CVPR'20) model using  "train_new_comb.py". 
```
    python train_new_comb.py  -train_batch_size 2  -category derain -exp_name DDN_SIRR_withGP  -lambda_GP 0.0015 -epoch_start 0 -version version1
```

