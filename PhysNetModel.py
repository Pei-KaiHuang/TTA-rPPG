import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from loss.ContrastLoss import ST_sampling, get_PSD_length

'''
Ref : https://github.com/ZitongYu/3DCDC-NAS
Spatio-Temporal Center-difference based Convolutional layer (3D version)
theta: control the percentage of original convolution and centeral-difference convolution
'''
class CDC_ST(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.01):

        super(CDC_ST, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, t, kernel_size, kernel_size] = self.conv.weight.shape

            # only CD works on temporal kernel size>1
            if self.conv.weight.shape[2] > 1:
                kernel_diff = self.conv.weight.sum(2).sum(2).sum(2)
                kernel_diff = kernel_diff[:, :, None, None, None]
                out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
                                    padding=0, dilation=self.conv.dilation, groups=self.conv.groups)
                return out_normal - self.theta * out_diff

            else:
                return out_normal
            
    
'''
Ref : https://github.com/ZitongYu/3DCDC-NAS
Temporal Center-difference based Convolutional layer (3D version)
theta: control the percentage of original convolution and centeral-difference convolution
'''
class CDC_T(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.01):

        super(CDC_T, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, t, kernel_size, kernel_size] = self.conv.weight.shape

            # only CD works on temporal kernel size > 1
            if self.conv.weight.shape[2] > 1:
                
                kernel_diff = self.conv.weight[:, :, 0, :, :].sum(2).sum(2) + self.conv.weight[:, :, 2, :, :].sum(2).sum(2)
                # torch.Size([in, out])
                
                kernel_diff = kernel_diff[:, :, None, None, None]
                # torch.Size([in, out, 1, 1, 1])
                
                out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
                                    padding=0, dilation=self.conv.dilation, groups=self.conv.groups)
                return out_normal - self.theta * out_diff

            else:
                return out_normal


'''
Ref : https://github.com/ZitongYu/3DCDC-NAS
Temporal Robust Center-difference based Convolutional layer (3D version)
theta: control the percentage of original convolution and centeral-difference convolution
'''
class CDC_TR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.1):

        super(CDC_TR, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.avgpool = nn.AvgPool3d(kernel_size=(kernel_size[0], 1, 1), stride=stride, padding=(padding[0], 0, 0))
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)
        local_avg = self.avgpool(x)
        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, t, kernel_size, kernel_size] = self.conv.weight.shape

            # only CD works on temporal kernel size>1
            if self.conv.weight.shape[2] > 1:
                kernel_diff = self.conv.weight[:, :, 0, :, :].sum(2).sum(2) + self.conv.weight[:, :, 2, :, :].sum(
                    2).sum(2)
                kernel_diff = kernel_diff[:, :, None, None, None]
                out_diff = F.conv3d(input=local_avg, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
                                    padding=0, groups=self.conv.groups)
                return out_normal - self.theta * out_diff

            else:
                return out_normal
                




'''
Ref : https://github.com/huiyu8794/LDCNet
Extended 3D version LDC
'''
class LDC_T(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        
        super(LDC_T, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
                
        self.learnable_theta = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)
        
        self.base_mask = nn.Parameter(torch.ones(self.conv.weight.size()), requires_grad=False)
        self.learnable_mask = nn.Parameter(torch.ones([self.conv.weight.size(0), self.conv.weight.size(1)]),
                                           requires_grad=True)
        # self.learnable_mask = nn.Parameter(torch.ones(self.conv.weight.size()), requires_grad=True)
                
        self.center_mask = torch.tensor([[[0, 0, 0], 
                                          [0, 0, 0], 
                                          [0, 0, 0]],
                                         [[0, 0, 0], 
                                          [0, 1, 0], 
                                          [0, 0, 0]],
                                         [[0, 0, 0], 
                                          [0, 0, 0], 
                                          [0, 0, 0]]]).cuda()

    def forward(self, x):
        
        kernel_diff = self.conv.weight[:, :, 0, :, :].sum(2).sum(2) + self.conv.weight[:, :, 2, :, :].sum(2).sum(2)
        kernel_diff = kernel_diff[:, :, None, None, None]
        
        h, w = self.conv.weight.size(3), self.conv.weight.size(4)
        center_weight = self.conv.weight[:, :, 1, h//2, w//2]
        
        mask = self.base_mask - self.learnable_theta * self.learnable_mask[:, :, None, None, None] * \
               self.center_mask * kernel_diff / center_weight[:, :, None, None, None]
               
        out_diff = F.conv3d(input=x, weight=self.conv.weight * mask, bias=self.conv.bias, stride=self.conv.stride,
                            padding=self.conv.padding,
                            groups=self.conv.groups)
        
        return out_diff


"""
Normal LDC
"""
class LDC_M(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        
        super(LDC_M, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
                
        self.learnable_theta = nn.Parameter(torch.ones(1) * 0.05, requires_grad=True)
        
        [_, _, t, h, w] = self.conv.weight.shape
        self.base_mask = nn.Parameter(torch.ones(self.conv.weight.size()), requires_grad=False)
        self.learnable_mask = nn.Parameter(torch.ones(self.conv.weight.size()), requires_grad=True)
        # self.learnable_mask = nn.Parameter(torch.ones([t, h, w]) * 0.1, requires_grad=True)
        
        
    def forward(self, x):
        
        # [_in, _out, _, _, _] = self.conv.weight.shape
        # ldp_mask = self.learnable_mask.expand(_in, _out, -1, -1, -1)
        mask = (1-self.learnable_theta)*self.base_mask + self.learnable_theta*self.learnable_mask
        
        out_diff = F.conv3d(input=x, weight=self.conv.weight * mask, bias=self.conv.bias, stride=self.conv.stride,
                            padding=self.conv.padding,
                            groups=self.conv.groups)
                
        return out_diff

# -------------------------------------------------------------------------------------------------------------------
# PhysNet model
# 
# the output is an ST-rPPG block rather than a rPPG signal.
# -------------------------------------------------------------------------------------------------------------------
class _PhysNet(nn.Module):
    def __init__(self, S=2, in_ch=3, conv3x3x3=nn.Conv3d):
        super().__init__()

        self.S = S  # S is the spatial dimension of ST-rPPG block

        self.encoder1_entangle = nn.Sequential(
            nn.Conv3d(in_channels=in_ch, out_channels=32, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ELU(),

            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        # encoder
        self.encoder2_entangle = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )

        # rPPG 
        self.encoder3_rPPG = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            conv3x3x3(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            conv3x3x3(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),

            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            conv3x3x3(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            conv3x3x3(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        # rPPG 
        self.encoder3_noise = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            conv3x3x3(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            conv3x3x3(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),

            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            conv3x3x3(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            conv3x3x3(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        # decoder to reach back initial temporal length
        self.decoder1_rPPG = nn.Sequential(
            conv3x3x3(in_channels=64, out_channels=64, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.decoder2_rPPG = nn.Sequential(
            conv3x3x3(in_channels=64, out_channels=64, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )
        self.end_rPPG = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, S, S)),
            nn.Conv3d(in_channels=64, out_channels=1, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        )

        # noise 
        self.decoder1_nosie = nn.Sequential(
            conv3x3x3(in_channels=64, out_channels=64, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.decoder2_noise = nn.Sequential(
            conv3x3x3(in_channels=64, out_channels=64, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )
        self.end_noise = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, S, S)),
            nn.Conv3d(in_channels=64, out_channels=1, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        )
 

    def forward(self, x, y=None):
        # x is fg, y is bg 
        if y is not None:

            means_x = torch.mean(x, dim=(2, 3, 4), keepdim=True)
            stds_x = torch.std(x, dim=(2, 3, 4), keepdim=True)
            x = (x - means_x) / stds_x  # (B, C, T, 128, 128)

            # nosie image, y 
            means_y = torch.mean(y, dim=(2, 3, 4), keepdim=True)
            stds_y = torch.std(y, dim=(2, 3, 4), keepdim=True)
            y = (y - means_y) / stds_y  # (B, C, T, 128, 128)

            parity_y = []
            y = self.encoder1_entangle(y)  # (B, C, T, 128, 128) 
            parity_y.append(y.size(2) % 2)
            y_entangle = self.encoder2_entangle(y)  # (B, 64, T/2, 32, 32)
            parity_y.append(y_entangle.size(2) % 2)
            y_noise = self.encoder3_noise(y_entangle)  # (B, 64, T/4, 16, 16)

            y = F.interpolate(y_noise, scale_factor=(2, 1, 1))  # (B, 64, T/2, 8, 8)
            y = self.decoder1_nosie(y)  # (B, 64, T/2, 8, 8)
            
            y_middle = y
            
            y = F.pad(y_middle, (0, 0, 0, 0, 0, parity_y[-1]), mode='replicate')
            y = F.interpolate(y, scale_factor=(2, 1, 1))  # (B, 64, T, 8, 8)
            y = self.decoder2_noise(y)  # (B, 64, T, 8, 8)
            y = F.pad(y, (0, 0, 0, 0, 0, parity_y[-2]), mode='replicate')
            y = self.end_noise(y)  # (B, 1, T, S, S), ST-rPPG block

            y_list = []
            for a in range(self.S):
                for b in range(self.S):
                    y_list.append(y[:, :, :, a, b])  # (B, 1, T)

            y = sum(y_list) / (self.S * self.S)  # (B, 1, T)
            Y = torch.cat(y_list + [y], 1)  # (B, N, T), flatten all spatial signals to the second dimension

            # rPPG image
            parity_x = []
            x = self.encoder1_entangle(x)  # (B, C, T, 128, 128) 
            parity_x.append(x.size(2) % 2)
            x_mix = self.encoder2_entangle(x)  # (B, 64, T/2, 32, 32)
            parity_x.append(x_mix.size(2) % 2)
            x_rPPG = self.encoder3_rPPG(x_mix)  # (B, 64, T/4, 16, 16)
            x_noise = self.encoder3_noise(x_mix) 

            x = F.interpolate(x_rPPG - x_noise, scale_factor=(2, 1, 1))  # (B, 64, T/2, 8, 8)
            x = self.decoder1_rPPG(x)  # (B, 64, T/2, 8, 8)
            x_middle = x
            x = F.pad(x_middle, (0, 0, 0, 0, 0, parity_x[-1]), mode='replicate')
            x = F.interpolate(x, scale_factor=(2, 1, 1))  # (B, 64, T, 8, 8)
            x = self.decoder2_rPPG(x)  # (B, 64, T, 8, 8)
            x = F.pad(x, (0, 0, 0, 0, 0, parity_x[-2]), mode='replicate')
            x = self.end_rPPG(x)  # (B, 1, T, S, S), ST-rPPG block

            x_list = []
            for a in range(self.S):
                for b in range(self.S):
                    x_list.append(x[:, :, :, a, b])  # (B, 1, T)

            x = sum(x_list) / (self.S * self.S)  # (B, 1, T)
            X = torch.cat(x_list + [x], 1)  # (B, N, T), flatten all spatial signals to the second dimension

            return X, Y, x_middle, y_middle
        else:
            means_x = torch.mean(x, dim=(2, 3, 4), keepdim=True)
            stds_x = torch.std(x, dim=(2, 3, 4), keepdim=True)
            x = (x - means_x) / stds_x  # (B, C, T, 128, 128)

            parity_x = []
            x = self.encoder1_entangle(x)  # (B, C, T, 128, 128) 
            parity_x.append(x.size(2) % 2)
            x_mix = self.encoder2_entangle(x)  # (B, 64, T/2, 32, 32)
            parity_x.append(x_mix.size(2) % 2)
            x_rPPG = self.encoder3_rPPG(x_mix)  # (B, 64, T/4, 16, 16)
            x_noise = self.encoder3_noise(x_mix)

            x = F.interpolate(x_rPPG - x_noise, scale_factor=(2, 1, 1))  # (B, 64, T/2, 8, 8)
            
            x_middle = x
            # print(x.shape)
            x = self.decoder1_rPPG(x)  # (B, 64, T/2, 8, 8)
            
            
            
            x = F.pad(x, (0, 0, 0, 0, 0, parity_x[-1]), mode='replicate')
            x = F.interpolate(x, scale_factor=(2, 1, 1))  # (B, 64, T, 8, 8)
            x = self.decoder2_rPPG(x)  # (B, 64, T, 8, 8)
            x = F.pad(x, (0, 0, 0, 0, 0, parity_x[-2]), mode='replicate')
            x = self.end_rPPG(x)  # (B, 1, T, S, S), ST-rPPG block
            
            

            x_list = []
            for a in range(self.S):
                for b in range(self.S):
                    x_list.append(x[:, :, :, a, b])  # (B, 1, T)

            x = sum(x_list) / (self.S * self.S)  # (B, 1, T)
            X = torch.cat(x_list + [x], 1)
            
            return X, x_middle
        
        
    def forward_cGAN(self, x):
    
        x = self.decoder1_rPPG(x)  # (B, 64, T/2, 8, 8)
        
        x = F.interpolate(x, scale_factor=(2, 1, 1))  # (B, 64, T, 8, 8)
        x = self.decoder2_rPPG(x)  # (B, 64, T, 8, 8)
        x = self.end_rPPG(x)  # (B, 1, T, S, S), ST-rPPG block
        
        

        x_list = []
        for a in range(self.S):
            for b in range(self.S):
                x_list.append(x[:, :, :, a, b])  # (B, 1, T)

        x = sum(x_list) / (self.S * self.S)  # (B, 1, T)
        X = torch.cat(x_list + [x], 1)
        return X


class PhysNet(nn.Module):
    def __init__(self, S=2, in_ch=3, conv_type=None, seq_len=300,
                 delta_t=300, numSample=1, class_num=2):
        super().__init__()
        
        # Ref : https://github.com/ZitongYu/3DCDC-NAS
        if conv_type == 'CDC_T':
            print('Using CDC_T convolutions')
            conv3x3x3 = CDC_T
        elif conv_type == 'CDC_ST':
            print('Using CDC_ST convolutions')
            conv3x3x3 = CDC_ST
        elif conv_type == 'CDC_TR':
            print('Using CDC_TR convolutions')
            conv3x3x3 = CDC_TR
        # Ref : https://github.com/huiyu8794/LDCNet
        elif conv_type == 'LDC_T':
            print('Using LDC_T convolutions')
            conv3x3x3 = LDC_T
        # Our
        elif conv_type == "LDC_M":
            print('Using LDC_M convolutions')
            conv3x3x3 = LDC_M
        else:
            print('Using vanilla 3D convolutions')
            conv3x3x3 = nn.Conv3d
        
        self.model = _PhysNet(S, in_ch, conv3x3x3)
        
        
    def forward(self, x, y=None, return_feature=False):
        
        if y is not None:
            rPPG_output, bg_output, x_middle, y_middle = self.model(x, y)

            if not return_feature:
                return rPPG_output, bg_output
            else:
                return rPPG_output, bg_output, x_middle, y_middle
        
        else:
            rPPG_output, x_middle = self.model(x)


            return rPPG_output, x_middle
        
    
    def forward_cGAN(self, x):
        
        rPPG_output = self.model.forward_cGAN(x)
        
        return rPPG_output



if __name__ == "__main__":
    
    B, S, T = 4, 2, 300
    x = torch.randn([B, 3, T, 64, 64])
    # y = torch.randn([B, 3, T, 64, 64])
    model = PhysNet(S=S, conv_type='LDC_M', seq_len=T, delta_t=T, numSample=1, class_num=2)
    # rPPG, bg = model(x, y)
    rPPG = model(x)
    print(rPPG.shape)

    # loss2 = BCE_loss(y_bg, bg_class_label[:, 0].long())
    
    # print(loss1, loss2)
    # print(loss1, loss2)

    # for conv in ['LDC_T', 'LDC_M', 'CDC_T', 'CDC_ST', 'CDC_TR', 'vanilla']:
    #     print(conv)
    #     model = PhysNet(conv_type=conv)
    #     model(x)
    #     # print(model(x))
    #     print(conv, 'done')
