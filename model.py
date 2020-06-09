import pdb
import math
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torchvision.models as models
from torch.autograd import Variable



def conv_block(in_dim,out_dim):
  return nn.Sequential(nn.Conv2d(in_dim,in_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.Conv2d(in_dim,in_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.Conv2d(in_dim,out_dim,kernel_size=1,stride=1,padding=0),
                       nn.AvgPool2d(kernel_size=2,stride=2))
def deconv_block(in_dim,out_dim):
  return nn.Sequential(nn.Conv2d(in_dim,out_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.Conv2d(out_dim,out_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.UpsamplingNearest2d(scale_factor=2))


class SEBlock(nn.Module):
    def __init__(self, input_dim, reduction):
        super().__init__()
        mid = int(input_dim / reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, reduction),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class Bottle2neckX(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, baseWidth, cardinality, stride=1, downsample=None, scale = 4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: base width.
            cardinality: num of convolution groups.
            stride: conv stride. Replaces pooling layer.
            scale: number of scale.
            type: 'normal': normal set. 'stage': frist block of a new stage.
        """
        super(Bottle2neckX, self).__init__()

        D = int(math.floor(planes * (baseWidth/64.0)))
        C = cardinality

        self.conv1 = nn.Conv2d(inplanes, D*C*scale, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(D*C*scale)
        self.SE = SEBlock(inplanes,C)

        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride = stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv2d(D*C, D*C, kernel_size=3, stride = stride, padding=1, groups=C, bias=False))
          bns.append(nn.BatchNorm2d(D*C))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(D*C*scale, inplanes  , kernel_size=1, stride=1, padding=0, bias=False)        
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.width  = D*C
        self.stype = stype
        self.scale = scale

    def forward(self, x):
        residual = x

        
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          
          sp = self.relu(self.bns[i](sp))
          sp = self.convs[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype=='normal':
          out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
          out = torch.cat((out, self.pool(spx[self.nums])),1)

        
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        #out = self.SE(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        #pdb.set_trace()
        out += residual
        

        return torch.cat([x, out], 1)

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(inter_planes)
        self.conv3 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)



class BottleneckBlock1(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock1, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=5, stride=1,
                               padding=2, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)



class BottleneckBlock2(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock2, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=7, stride=1,
                               padding=3, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.upsample_nearest(out, scale_factor=2)



class TransitionBlock1(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock1, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)



class TransitionBlock3(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock3, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out






class vgg19ca(nn.Module):
    def __init__(self):
        super(vgg19ca, self).__init__()




        ############# 256-256  ##############
        haze_class = models.vgg19_bn(pretrained=True)
        self.feature = nn.Sequential(haze_class.features[0])

        for i in range(1,3):
            self.feature.add_module(str(i),haze_class.features[i])

        self.conv16=nn.Conv2d(64, 24, kernel_size=3,stride=1,padding=1)  # 1mm
        self.dense_classifier=nn.Linear(127896, 512)
        self.dense_classifier1=nn.Linear(512, 4)


    def forward(self, x):

        feature=self.feature(x)
        # feature = Variable(feature.data, requires_grad=True)

        feature=self.conv16(feature)
        # print feature.size()

        # feature=Variable(feature.data,requires_grad=True)



        out = F.relu(feature, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7).view(out.size(0), -1)
        # print out.size()

        # out=Variable(out.data,requires_grad=True)
        out = F.relu(self.dense_classifier(out))
        out = (self.dense_classifier1(out))


        return out


class scale_residue_est(nn.Module):
    def __init__(self):
        super(scale_residue_est, self).__init__()

        self.conv1 = BottleneckBlock(64, 32)
        self.trans_block1 = TransitionBlock3(96, 32)
        self.conv2 = BottleneckBlock(32, 32)
        self.trans_block2 = TransitionBlock3(64, 32)
        self.conv3 = BottleneckBlock(32, 32)
        self.trans_block3 = TransitionBlock3(64, 32)
        self.conv_refin = nn.Conv2d(32, 16, 3, 1, 1)
        self.tanh = nn.Tanh()
        self.refine3 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1=self.conv1(x)
        x1 = self.trans_block1(x1)
        x2=self.conv2(x1)
        x2 = self.trans_block2(x2)
        x3=self.conv3(x2)
        x3 = self.trans_block3(x3)
        x4 = self.relu((self.conv_refin(x3)))
        residual = self.tanh(self.refine3(x4))

        return residual

class scale_residue_conf(nn.Module):
    def __init__(self):
        super(scale_residue_conf, self).__init__()

        self.conv1 = nn.Conv2d(35,16,3,1,1)#BottleneckBlock(35, 16)
        #self.trans_block1 = TransitionBlock3(51, 8)
        self.conv2 = BottleneckBlock(16, 16)
        self.trans_block2 = TransitionBlock3(32, 16)
        self.conv3 = BottleneckBlock(16, 16)
        self.trans_block3 = TransitionBlock3(32, 16)
        self.conv_refin = nn.Conv2d(16, 16, 3, 1, 1)
        self.sig = torch.nn.Sigmoid()
        self.refine3 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1=self.conv1(x)
        #x1 = self.trans_block1(x1)
        x2=self.conv2(x1)
        x2 = self.trans_block2(x2)
        x3=self.conv3(x2)
        x3 = self.trans_block3(x3)
        residual = self.sig(self.refine3(x3))

        return residual



class DeRain_v1(nn.Module):
    def __init__(self):
        super(DeRain_v1, self).__init__()
        self.baseWidth = 12#4#16
        self.cardinality = 8#8#16
        self.scale = 6#4#5
        self.stride = 1
        ############# Block1-scale 1.0  ##############
        self.conv_input=nn.Conv2d(3,16,3,1,1)
        self.dense_block1=Bottle2neckX(16,16, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')

        ############# Block2-scale 0.50  ##############
        self.trans_block2=TransitionBlock1(32,32)
        self.dense_block2=Bottle2neckX(32,32, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        self.trans_block2_o=TransitionBlock3(64,32)

        ############# Block3-scale 0.250  ##############
        self.trans_block3=TransitionBlock1(32,32)
        self.dense_block3=Bottle2neckX(32,32, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        self.trans_block3_o=TransitionBlock3(64,32)

        ############# Block4-scale 0.25  ##############
        self.trans_block4=TransitionBlock1(32,64)
        self.dense_block4=Bottle2neckX(64,64, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        self.trans_block4_o=TransitionBlock3(128,64)

        ############# Block5-scale 0.25  ##############
        self.dense_block5=Bottle2neckX(64,64, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        self.trans_block5_o=TransitionBlock3(128,64)

        ############# Block6-scale 0.25  ##############
        self.dense_block6=Bottle2neckX(64,64, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        self.trans_block6_o=TransitionBlock3(128,64)

        ############# Block7-scale 0.25  ############## 7--3 skip connection
        self.trans_block7=TransitionBlock(64,32)
        self.dense_block7=Bottle2neckX(64,64, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        self.trans_block7_o=TransitionBlock3(128,32)

        ############# Block8-scale 0.5  ############## 8--2 skip connection
        self.trans_block8=TransitionBlock(32,32)
        self.dense_block8=Bottle2neckX(64,64, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        self.trans_block8_o=TransitionBlock3(128,32)

        ############# Block9-scale 1.0  ############## 9--1 skip connection
        self.trans_block9=TransitionBlock(32,32)
        self.dense_block9=Bottle2neckX(80,80, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        self.trans_block9_o=TransitionBlock3(160,16)


        self.conv_refin=nn.Conv2d(16,16,3,1,1)
        self.tanh=nn.Tanh()


        self.refine3= nn.Conv2d(16, 3, kernel_size=3,stride=1,padding=1)
        self.zout= nn.Conv2d(64, 32, kernel_size=3,stride=1,padding=1)
        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)
        self.refineclean1= nn.Conv2d(3, 8, kernel_size=7,stride=1,padding=3)
        self.refineclean2= nn.Conv2d(8, 3, kernel_size=3,stride=1,padding=1)



    def forward(self, xin):
        x= self.conv_input(xin)
        #Size - 1.0
        x1=(self.dense_block1(x))

        #Size - 0.5        
        x2_i=self.trans_block2(x1)
        x2_i=self.dense_block2(x2_i)
        x2=self.trans_block2_o(x2_i)

        #Size - 0.25
        x3_i=self.trans_block3(x2)
        x3_i=self.dense_block3(x3_i)
        x3=self.trans_block3_o(x3_i)

        #Size - 0.125
        x4_i=self.trans_block4(x3)
        x4_i=self.dense_block4(x4_i)
        x4=self.trans_block4_o(x4_i)

        x5_i=self.dense_block5(x4)
        x5=self.trans_block5_o(x5_i)

        x6_i=self.dense_block6(x5)
        x6=self.trans_block6_o(x6_i)
        z = self.zout(x6)

        x7_i=self.trans_block7(x6)
        # print(x4.size())
        # print(x7_i.size())
        x7_i=self.dense_block7(torch.cat([x7_i, x3], 1))
        x7=self.trans_block7_o(x7_i)

        x8_i=self.trans_block8(x7)
        x8_i=self.dense_block8(torch.cat([x8_i, x2], 1))
        x8=self.trans_block8_o(x8_i)

        x9_i=self.trans_block9(x8)
        x9_i=self.dense_block9(torch.cat([x9_i, x1,x], 1))
        x9=self.trans_block9_o(x9_i)

        # x10_i=self.trans_block10(x9)
        # x10_i=self.dense_block10(torch.cat([x10_i, x1,x], 1))
        # x10=self.trans_block10_o(x10_i)

        x11=x-self.relu((self.conv_refin(x9)))
        residual=self.tanh(self.refine3(x11))
        clean = residual
        clean=self.relu(self.refineclean1(clean))
        clean=self.tanh(self.refineclean2(clean))

        return clean,z

class DeRain_v2(nn.Module):
    def __init__(self):
        super(DeRain_v2, self).__init__()
        self.baseWidth = 12#4#16
        self.cardinality = 8#8#16
        self.scale = 6#4#5
        self.stride = 1
        ############# Block1-scale 1.0  ##############
        self.conv_input=nn.Conv2d(3,16,3,1,1)
        self.dense_block1=Bottle2neckX(16,16, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')

        ############# Block2-scale 0.50  ##############
        self.trans_block2=TransitionBlock1(32,32)
        self.dense_block2=Bottle2neckX(32,32, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        self.trans_block2_o=TransitionBlock3(64,32)

        ############# Block3-scale 0.250  ##############
        self.trans_block3=TransitionBlock1(32,32)
        self.dense_block3=Bottle2neckX(32,32, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        self.trans_block3_o=TransitionBlock3(64,64)

        ############# Block4-scale 0.25  ##############
        self.trans_block4=TransitionBlock1(64,128)
        self.dense_block4=Bottle2neckX(128,128, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        self.trans_block4_o=TransitionBlock3(256,128)

        ############# Block5-scale 0.25  ##############
        self.dense_block5=Bottle2neckX(128,128, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        self.trans_block5_o=TransitionBlock3(256,128)

        ############# Block6-scale 0.25  ##############
        self.dense_block6=Bottle2neckX(128,128, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        self.trans_block6_o=TransitionBlock3(256,128)

        ############# Block7-scale 0.25  ############## 7--3 skip connection
        self.trans_block7=TransitionBlock(128,64)
        self.dense_block7=Bottle2neckX(128,128, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        self.trans_block7_o=TransitionBlock3(256,32)

        ############# Block8-scale 0.5  ############## 8--2 skip connection
        self.trans_block8=TransitionBlock(32,32)
        self.dense_block8=Bottle2neckX(64,64, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        self.trans_block8_o=TransitionBlock3(128,32)

        ############# Block9-scale 1.0  ############## 9--1 skip connection
        self.trans_block9=TransitionBlock(32,32)
        self.dense_block9=Bottle2neckX(80,80, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        self.trans_block9_o=TransitionBlock3(160,16)


        self.conv_refin=nn.Conv2d(16,16,3,1,1)
        self.tanh=nn.Tanh()
        self.sig=nn.Sigmoid()


        self.refine3= nn.Conv2d(16, 3, kernel_size=3,stride=1,padding=1)
        self.zout= nn.Conv2d(128, 32, kernel_size=3,stride=1,padding=1)
        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)
        self.refineclean1= nn.Conv2d(3, 8, kernel_size=7,stride=1,padding=3)
        self.refineclean2= nn.Conv2d(8, 3, kernel_size=3,stride=1,padding=1)



    def forward(self, xin):
        x= self.conv_input(xin)
        #Size - 1.0
        x1=(self.dense_block1(x))

        #Size - 0.5        
        x2_i=self.trans_block2(x1)
        x2_i=self.dense_block2(x2_i)
        x2=self.trans_block2_o(x2_i)

        #Size - 0.25
        x3_i=self.trans_block3(x2)
        x3_i=self.dense_block3(x3_i)
        x3=self.trans_block3_o(x3_i)

        #Size - 0.125
        x4_i=self.trans_block4(x3)
        x4_i=self.dense_block4(x4_i)
        x4=self.trans_block4_o(x4_i)

        x5_i=self.dense_block5(x4)
        x5=self.trans_block5_o(x5_i)

        x6_i=self.dense_block6(x5)
        x6=self.trans_block6_o(x6_i)
        z = self.zout(x6)

        x7_i=self.trans_block7(x6)
        # print(x4.size())
        # print(x7_i.size())
        x7_i=self.dense_block7(torch.cat([x7_i, x3], 1))
        x7=self.trans_block7_o(x7_i)

        x8_i=self.trans_block8(x7)
        x8_i=self.dense_block8(torch.cat([x8_i, x2], 1))
        x8=self.trans_block8_o(x8_i)

        x9_i=self.trans_block9(x8)
        x9_i=self.dense_block9(torch.cat([x9_i, x1,x], 1))
        x9=self.trans_block9_o(x9_i)

        # x10_i=self.trans_block10(x9)
        # x10_i=self.dense_block10(torch.cat([x10_i, x1,x], 1))
        # x10=self.trans_block10_o(x10_i)

        x11=x-self.relu((self.conv_refin(x9)))
        residual=self.tanh(self.refine3(x11))
        clean = residual
        clean=self.relu(self.refineclean1(clean))
        clean=self.sig(self.refineclean2(clean))

        return clean,z

def gradient(y):
    gradient_h=y[:, :, :, :-1] - y[:, :, :, 1:]
    gradient_v=y[:, :, :-1, :] - y[:, :, 1:, :]

    return gradient_h, gradient_v

def TV(y):
    gradient_h=torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])
    gradient_v=torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])

    return gradient_h, gradient_v

class discriminator(nn.Module):
    # initializers
    def __init__(self, d=64):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(6, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()