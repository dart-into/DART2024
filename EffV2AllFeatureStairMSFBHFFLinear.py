import torch
import torch.nn as nn
import torch.nn.functional as F
# from efficientnet_pytorch import EfficientNet
import math
import timm
#from einops import rearrange

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        # padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        # self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CSAB(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CSAB, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out




class HyperStructure2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HyperStructure2, self).__init__()
        self.hyper_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.hyper_block(x)

class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        pretrained_cfg_overlay = {'file': r"/home/user/DART2024-main/tf_efficientnetv2_s_21k-6337ad01.pth"}
        self.RGBNet = timm.create_model('tf_efficientnetv2_s.in21k', pretrained_cfg_overlay=pretrained_cfg_overlay,
                                        features_only=True, pretrained=True)
        self.GraNet = timm.create_model('tf_efficientnetv2_s.in21k', pretrained_cfg_overlay=pretrained_cfg_overlay,
                                        features_only=True, pretrained=True)
     

        self.HFF_dp = 0

        self.gap = nn.AdaptiveAvgPool2d((10, 10))
        self.gap1 = nn.AdaptiveAvgPool2d((1, 1))

        self.msfb1 = MSFB(inchannel=256, outchannel=128)
        self.msfb2 = MSFB(inchannel=256, outchannel=128)
        self.fu1 = HFF_block(ch_1=512, ch_2=512, last_ch=48, ch_int=512, ch_out=1024, drop_rate=self.HFF_dp)
        ##rgb
        self.hyper0_1_rgb = HyperStructure2(24, 48) #112->56
        self.hyper0_2_rgb = HyperStructure2(48, 64) #56->28
        self.hyper0_3_rgb = HyperStructure2(64, 160) #28->14
        self.hyper0_4_rgb = HyperStructure2(160, 256) #14->7

        self.hyper1_1_rgb = HyperStructure2(48, 64) #56->28
        self.hyper1_2_rgb = HyperStructure2(64, 160) #28->14
        self.hyper1_3_rgb = HyperStructure2(160, 256) #14->7

        self.hyper2_1_rgb = HyperStructure2(64, 160) #28->14
        self.hyper2_2_rgb = HyperStructure2(160, 256) #14->7

        self.hyper3_1_rgb = HyperStructure2(160, 256) #14->7
        ##gra
        
        self.hyper0_1_gra = HyperStructure2(24, 48)  # 112->56
        self.hyper0_2_gra = HyperStructure2(48, 64)  # 56->28
        self.hyper0_3_gra = HyperStructure2(64, 160)  # 28->14
        self.hyper0_4_gra = HyperStructure2(160, 256)  # 14->7

        self.hyper1_1_gra = HyperStructure2(48, 64)  # 56->28
        self.hyper1_2_gra = HyperStructure2(64, 160)  # 28->14
        self.hyper1_3_gra = HyperStructure2(160, 256)  # 14->7

        self.hyper2_1_gra = HyperStructure2(64, 160)  # 28->14
        self.hyper2_2_gra = HyperStructure2(160, 256)  # 14->7

        self.hyper3_1_gra = HyperStructure2(160, 256)  # 14->7


    def forward(self, input1, input2):
        input_rgb = input1.view(-1, input1.size(-3), input1.size(-2), input1.size(-1))
        input_gra = input2.view(-1, input2.size(-3), input2.size(-2), input2.size(-1))
        # print('input', input.size())  # [16, 3, 224, 224]

        endpoints_rgb = self.RGBNet(input_rgb)
        endpoints_gra = self.GraNet(input_gra)

        # print(endfeature_rgb.shape)
        ## Feature Compute
        a0_rgb = endpoints_rgb[0]  # [1, 24, 112, 112]
        a1_rgb = endpoints_rgb[1]  # [1, 48, 56, 56]
        a2_rgb = endpoints_rgb[2]  # [1, 64, 28, 28]
        a3_rgb = endpoints_rgb[3]  # [1, 160, 14, 14]
        a4_rgb = endpoints_rgb[4]  # [1, 256, 7, 7]
       
        a0_gra = endpoints_rgb[0]  # [1, 24, 112, 112]
        a1_gra = endpoints_gra[1]  # [1, 48, 56, 56]
        a2_gra = endpoints_gra[2]  # [1, 64, 28, 28]
        a3_gra = endpoints_gra[3]  # [1, 160, 14, 14]
        a4_gra = endpoints_gra[4]  # [1, 256, 7, 7]
        
     
        rgb_hyper0 = self.hyper0_1_rgb(a0_rgb)  # [1, 48, 56, 56]
        
        rgb_hyper0 = self.hyper0_2_rgb(rgb_hyper0+a1_rgb)  # [1, 48, 28, 28]
        rgb_hyper1 = self.hyper1_1_rgb(a1_rgb)  # [1, 64, 28, 28]
        
        rgb_hyper0 = self.hyper0_3_rgb(rgb_hyper0+a2_rgb) # [1, 160, 14, 14]
        rgb_hyper1 = self.hyper1_2_rgb(rgb_hyper1 + a2_rgb)  # [1, 160, 14, 14]
        rgb_hyper2 = self.hyper2_1_rgb(a2_rgb)  # [1, 160, 14, 14]
        
        rgb_hyper0 = self.hyper0_4_rgb(rgb_hyper0 + a3_rgb) # [1, 256, 7, 7]
        rgb_hyper1 = self.hyper1_3_rgb(rgb_hyper1 + a3_rgb)  # [1, 256, 7, 7]
        rgb_hyper2 = self.hyper2_2_rgb(rgb_hyper2 + a3_rgb) # [1, 256, 7, 7]
        rgb_hyper3 = self.hyper3_1_rgb(a3_rgb)  # [1, 256, 7, 7]
        
        RGB_combined = rgb_hyper0 + rgb_hyper1 + rgb_hyper2 + rgb_hyper3 + a4_rgb  # [1, 256, 7, 7]
        ###################Gra
        gra_hyper0 = self.hyper0_1_gra(a0_gra)  # [1, 48, 56, 56]
        
        gra_hyper0 = self.hyper0_2_gra(gra_hyper0 + a1_gra)  # [1, 64, 28, 28]
        gra_hyper1 = self.hyper1_1_gra(a1_gra)  # [1, 64, 28, 28]
        
        gra_hyper0 = self.hyper0_3_gra(gra_hyper0 + a2_gra)  # [1, 160, 14, 14]
        gra_hyper1 = self.hyper1_2_gra(gra_hyper1 + a2_gra)  # [1, 160, 14, 14]
        gra_hyper2 = self.hyper2_1_gra(a2_gra)  # [1, 160, 14, 14]
        
        gra_hyper0 = self.hyper0_4_gra(gra_hyper0 + a3_gra)  # [1, 256, 7, 7]
        gra_hyper1 = self.hyper1_3_gra(gra_hyper1 + a3_gra)  # [1, 256, 7, 7]
        gra_hyper2 = self.hyper2_2_gra(gra_hyper2 + a3_gra)  # [1, 256, 7, 7]
        gra_hyper3 = self.hyper3_1_gra(a3_gra)  # [1, 256, 7, 7]

      
        Gra_combined =gra_hyper0 + gra_hyper1 + gra_hyper2 + gra_hyper3 + a4_gra # [1, 256, 7, 7]
      
        RGB_Feature = self.msfb1(RGB_combined)  # [1, 512, 4, 4]
        Gra_Feature = self.msfb2(Gra_combined)  # [1, 512, 4, 4]


        x_f_1 = self.fu1(RGB_Feature, Gra_Feature, None)
        f1 = self.gap1(x_f_1)  # [1,512,1,1]
       
        f1 = f1.view(f1.size(0), -1)  # [1,512]
      
        return f1


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# Hierachical Feature Fusion Block
class HFF_block(nn.Module):
    def __init__(self, ch_1, ch_2, last_ch, ch_int, ch_out, drop_rate=0.):
        super(HFF_block, self).__init__()

        self.CSAB_RGB = CSAB(ch_1)
        self.CSAB_Gra = CSAB(ch_2)
        self.W_l = Conv(ch_1, ch_int, 1, bn=False, relu=False)  # 起一个bn作用
        self.W_g = Conv(ch_2, ch_int, 1, bn=False, relu=False)
        self.Avg = nn.AvgPool2d(2, stride=2)
        self.Updim = Conv(last_ch, ch_int, 1, bn=False, relu=True)

       
        self.W3 = Conv(ch_int * 3, ch_int, 1, bn=False, relu=False)
        self.W = Conv(ch_int * 2, ch_int, 1, bn=False, relu=False)

        self.gelu = nn.GELU()

        self.residual = IRMLP(ch_1 + ch_2 + ch_int, ch_out)
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, l, g, f):

        W_local = self.W_l(l)  
        W_global = self.W_g(g)  
        if f is not None:
            W_f = self.Updim(f)
            W_f = self.Avg(W_f)
            shortcut = W_f
            X_f = torch.cat([W_f, W_local, W_global], 1)
          
            X_f = self.W3(X_f)
            X_f = self.gelu(X_f)
        else:
            shortcut = 0
            X_f = torch.cat([W_local, W_global], 1)  
           
            X_f = self.W(X_f)  # 1*1 conv #[96*56*56]
            X_f = self.gelu(X_f)  # Gelu

       

        l = self.CSAB_RGB(l)
      
        g = self.CSAB_Gra(g)

        fuse = torch.cat([g, l, X_f], 1)
        #fuse = self.norm3(fuse)
        fuse = self.residual(fuse)
        fuse = shortcut + self.drop_path(fuse)
        return fuse



def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


#### Inverted Residual MLP
class IRMLP(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(IRMLP, self).__init__()
        self.conv1 = Conv(inp_dim, inp_dim, 3, relu=False, bias=False, group=inp_dim)
        self.conv2 = Conv(inp_dim, inp_dim * 2, 1, relu=False, bias=False)
        self.conv3 = Conv(inp_dim * 2, out_dim, 1, relu=False, bias=False, bn=True)
        self.gelu = nn.GELU()
      

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.gelu(out)
        out += residual

       
        out = self.conv2(out)
        out = self.gelu(out)
        out = self.conv3(out)

        return out


#### Conv ####
class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True, group=1):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class MSFB(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(MSFB, self).__init__()

        self.conv1_MSFB_c1s2 = nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=1, stride=2,
                                         padding=0)  # 第一流
        self.conv3_MSFB_c3s1 = nn.Conv2d(in_channels=outchannel, out_channels=outchannel, kernel_size=3, stride=1,
                                         padding=1)  # 第三流第二个
        self.conv3_MSFB_c3s1_2 = nn.Conv2d(in_channels=outchannel, out_channels=outchannel, kernel_size=3, stride=1,
                                           padding=1)  # 第四流第二个
        self.conv3_MSFB_c3s1_3 = nn.Conv2d(in_channels=outchannel, out_channels=outchannel, kernel_size=3, stride=1,
                                           padding=1)  # 第四流第三个
        self.conv3_MSFB_c3s2 = nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=3, stride=2,
                                         padding=1)  # 第二流
        self.conv3_MSFB_c3s2_2 = nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=3, stride=2,
                                           padding=1)  # 第三流第一个
        self.conv3_MSFB_c3s2_3 = nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=3, stride=2,
                                           padding=1)  # 第四流第一个

    def forward(self, x):
        x1 = self.conv1_MSFB_c1s2(x)
        x3 = self.conv3_MSFB_c3s2(x)
        x5 = self.conv3_MSFB_c3s1(self.conv3_MSFB_c3s2_2(x))
        x7 = self.conv3_MSFB_c3s1_3(self.conv3_MSFB_c3s1_2(self.conv3_MSFB_c3s2_3(x)))

        output = torch.cat((x1, x3, x5, x7), 1)
        return output


class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(1024, 512)  
        self.fc2 = nn.Linear(512, 1)  
       

    
    def forward(self, x1):
       

        x1 = self.fc1(x1)

        out = self.fc2(x1)
      
        return out



class Net(nn.Module):
    def __init__(self, headnet, net):
        super(Net, self).__init__()
        self.headnet = headnet
        self.net = net

    def forward(self, x1, x2):
        f1 = self.headnet(x1, x2)
        output = self.net(f1)
        return output
# if __name__ == '__main__':
#net1 = FeatureNet()
#net2 = FCNet()
#model = Net(headnet=net1, net=net2)
#total_params = sum(p.numel() for p in model.parameters())
#print(f"Total parameters: {total_params}")
#input = torch.randn((2, 3, 224, 224))
# net = DACNN().cuda()
# input = torch.tensor(torch.randn((1, 3, 224, 224))).cuda()
#output = model(input,input)
#print(output)
