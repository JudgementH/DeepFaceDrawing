import torch
import torch.nn as nn
from torch.nn import init
# import torchsnooper
import functools
from torch import autograd
from torch.autograd import Variable
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torchvision.models


#####################################################################
#####   损失函数
#####################################################################
class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()
        self.criterion = nn.BCELoss()

    def forward(self, pred, label):
        return self.criterion(pred, label)


class HingeLoss(nn.Module):
    """
    论文地址：https://arxiv.org/pdf/1705.02894.pdf
    """

    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, pred, label: bool):
        if label:
            loss = F.relu(1 - pred).mean()
        else:
            loss = F.relu(1 + pred).mean()
        return loss


class WGANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(WGANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label

    def forward(self, input, target_is_real: bool):
        if target_is_real:
            loss = input.mean()
        else:
            loss = -input.mean()
        return loss


#####################################################################
#####   函数
#####################################################################
def get_norm_layer(planes, norm_type='batch', num_groups=4):
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d(planes)
    elif norm_type == 'instance':
        norm_layer = nn.InstanceNorm2d(planes)
    elif norm_type == 'group':
        norm_layer = nn.GroupNorm(num_groups, planes)
    else:
        raise NotImplementedError('normalization [%s] 不合法' % norm_type)
    return norm_layer

def get_scheduler(optimizer, opt):
    """
    调整优化器的学习率
    Parameters:
        optimizer          -- 优化器的名称
        opt (option class) -- 是base_option的子类，需要有属性　
                                opt.lr_policy linear | step | plateau | cosine
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def adain(content_feat, style_feat):
    """论文地址 https://arxiv.org/pdf/1703.06868.pdf
    :param content_feat: content特征
    :param style_feat: feat特征
    :return feature (tensor[batch_size, channel, size, size]): feature_map with content and style
    """
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def calc_mean_std(feat, eps=1e-5):
    """
    通过feature map的均值和标准差
    :param feat: feature map with size [batch, channel, size, size]
    :param eps: 防止分母为0
    :return:
        feat_mean (tensor [batch, channel, 1, 1]): mean
        feat_std (tensor [batch, channel, 1, 1]): std
    """
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def get_batched_gram_matrix(input):
    """
    返回tensor的Gram matrix
    :param input (tensor [B,C,H,W]):
    :return gram (tensor [B,C,C]):
    """
    a, b, c, d = input.size()
    features = input.view(a, b, c * d)
    G = torch.bmm(features, features.transpose(2, 1))
    return G.div(b * c * d)


def weights_init_normal(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1, 0.02)
        nn.init.constant_(m.bias, 0.0)

def init_weights(net, init_type='normal', init_gain=0.02):
    """
    初始化网络权重.
    Parameters:
        net (network)   -- 要被初始化的网络
        init_type (str) -- 初始化选项: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- 缩放因子

    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


#####################################################################
#####   组件
#####################################################################
class ResBlock(nn.Module):
    """残差网络块
    """

    def __init__(self, in_channels: int, norm_type: str = 'batch', padding_type='zero', dropout=0.0,
                 use_bias=False):
        """论文地址：https://arxiv.org/pdf/1512.03385.pdf
        :param in_channels (int): 输入通道数
        :param out_channels: 输出通道数
        :param norm_layer: normalization的方式，可选为: batch | instance
        :param padding_type: padding的方式，可选为: reflect| replicate| zero
        :param dropout: dropout概率
        :param use_bias: 偏置
        """
        super(ResBlock, self).__init__()
        model = []
        p = 0
        if padding_type == 'reflect':
            model += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            model += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] 不合法' % padding_type)

        model += [nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=p, bias=use_bias),
                  get_norm_layer(in_channels, norm_type=norm_type),
                  nn.ReLU(True)]
        if dropout > 0:
            model += [nn.Dropout(dropout)]

        p = 0
        if padding_type == 'reflect':
            model += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            model += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] 不合法' % padding_type)

        model += [nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=p, bias=use_bias),
                  get_norm_layer(in_channels, norm_type=norm_type)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x) + x
        return out


class UpConvBlock(nn.Module):
    """图片大小变为原来的2倍"""

    def __init__(self, in_channel, out_channel, norm_layer_type='batch'):
        """
        :param in_channel: 输入通道数
        :param out_channel: 输出通道数
        :param norm_layer_type: normalization的方式，可选为: batch | instance
        """
        super(UpConvBlock, self).__init__()
        self.main = nn.Sequential(
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channel, out_channel, 3, 1, 0, bias=True)),
            get_norm_layer(out_channel, norm_layer_type),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x):
        y = F.interpolate(x, scale_factor=2)
        return self.main(y)


class DownConvBlock(nn.Module):
    """图片大小变为原来的1/2"""

    def __init__(self, in_channel, out_channel, norm_layer_type='batch', down=True):
        """
        :param in_channel: 输入通道数
        :param out_channel: 输出通道数
        :param norm_layer_type: normalization的方式，可选为: batch | instance
        """
        super(DownConvBlock, self).__init__()

        m = [spectral_norm(nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=True)),
             get_norm_layer(out_channel, norm_layer_type),
             nn.LeakyReLU(0.1)]
        if down:
            m.append(nn.AvgPool2d(2, 2))
        self.main = nn.Sequential(*m)

    def forward(self, x):
        return self.main(x)


class Adaptive_pool(nn.Module):
    '''
    输入tensor大小 B x C' X C'
    通过池化，输出tensor大小 B x C x H x W
    '''

    def __init__(self, channel_out, hw_out):
        super(Adaptive_pool, self).__init__()
        self.channel_out = channel_out
        self.hw_out = hw_out
        self.pool = nn.AdaptiveAvgPool2d((channel_out, hw_out ** 2))

    def forward(self, input):
        if len(input.shape) == 3:
            input.unsqueeze_(1)
        return self.pool(input).view(-1, self.channel_out, self.hw_out, self.hw_out)


class EncoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=4, stride=1, padding=2):
        """
        大小默认变为原本的1/2
        :param in_channel:
        :param out_channel:
        :param kernel_size:
        :param stride:
        :param padding:
        """
        super(EncoderBlock, self).__init__()
        # convolution to halve the dimensions
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channel, momentum=0.9)
        self.relu = nn.LeakyReLU(1)

    def forward(self, input, out=False):
        if out:
            input = self.conv(input)
            ten_out = input
            input = self.bn(input)
            input = self.relu(input)
            return (input, ten_out)
        else:
            input = self.conv(input)
            input = self.bn(input)
            input = self.relu(input)
            return input


class DecoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=4, padding=1, stride=2, output_padding=0, relu=True):
        """
        大小变换原来的2倍
        :param in_channel:
        :param out_channel:
        :param kernel_size:
        :param padding:
        :param stride:
        :param output_padding:
        :param relu:
        """
        super(DecoderBlock, self).__init__()
        layers_list = []
        layers_list += [nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride,
                                           output_padding=output_padding),
                        nn.BatchNorm2d(out_channel, momentum=0.9)]

        if (relu == True):
            layers_list += [nn.LeakyReLU(1)]
        self.conv = nn.Sequential(*layers_list)

    def forward(self, input):
        out = self.conv(input)
        return out


class UpConvResBlock(nn.Module):
    def conv3x3(self, inplanes, out_planes, stride=1, use_sn=True):
        if use_sn:
            return spectral_norm(nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1))
        else:
            return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1)

    def __init__(self, inplanes, planes, stride=1, dropout=0.0, use_sn=False, norm_layer='batch', num_groups=8):
        super(UpConvResBlock, self).__init__()
        model = []
        model += upsampleLayer(inplanes, planes, upsample='nearest', use_sn=use_sn)
        if norm_layer != 'none':
            model += [get_norm_layer(planes, norm_layer, num_groups)]  # [nn.BatchNorm2d(planes)]
        model += [nn.ReLU(inplace=True)]
        model += [self.conv3x3(planes, planes, stride, use_sn)]
        if norm_layer != 'none':
            model += [get_norm_layer(planes, norm_layer, num_groups)]  # [nn.BatchNorm2d(planes)]
        model += [nn.ReLU(inplace=True)]
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)

        residual_block = []
        residual_block += upsampleLayer(inplanes, planes, upsample='bilinear', use_sn=use_sn)
        self.residual_block = nn.Sequential(*residual_block)

    def forward(self, x):
        residual = self.residual_block(x)
        out = self.model(x)
        out += residual
        return out


class DownConvResBlock(nn.Module):
    def conv3x3(self, inplanes, out_planes, stride=1, use_sn=True):
        if use_sn:
            return spectral_norm(nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1))
        else:
            return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1)

    def __init__(self, inplanes, planes, stride=1, dropout=0.0, use_sn=False, norm_layer='batch', num_groups=8):
        super(DownConvResBlock, self).__init__()
        model = []
        model += downsampleLayer(inplanes, planes, downsample='avgpool', use_sn=use_sn)
        if norm_layer != 'none':
            model += [get_norm_layer(planes, norm_layer, num_groups)]  # [nn.BatchNorm2d(planes)]
        model += [nn.ReLU(inplace=True)]
        model += [self.conv3x3(planes, planes, stride, use_sn)]
        if norm_layer != 'none':
            model += [get_norm_layer(planes, norm_layer, num_groups)]  # [nn.BatchNorm2d(planes)]
        model += [nn.ReLU(inplace=True)]
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)

        residual_block = []
        residual_block += downsampleLayer(inplanes, planes, downsample='avgpool', use_sn=use_sn)
        self.residual_block = nn.Sequential(*residual_block)

    def forward(self, x):
        residual = self.residual_block(x)
        out = self.model(x)
        out += residual
        return out


def downsampleLayer(inplanes, outplanes, downsample='basic', use_sn=True):
    # padding_type = 'zero'
    if downsample == 'basic' and not use_sn:
        downconv = [nn.Conv2d(
            inplanes, outplanes, kernel_size=3, stride=2, padding=1)]
    elif downsample == 'avgpool' and not use_sn:
        downconv = [nn.AvgPool2d(2, stride=2),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0)]
    elif downsample == 'maxpool' and not use_sn:
        downconv = [nn.MaxPool2d(2, stride=2),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0)]

    elif downsample == 'basic' and use_sn:
        downconv = [spectral_norm(nn.Conv2d(
            inplanes, outplanes, kernel_size=3, stride=2, padding=1))]
    elif downsample == 'avgpool' and use_sn:
        downconv = [nn.AvgPool2d(2, stride=2),
                    nn.ReflectionPad2d(1),
                    spectral_norm(nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0))]
    elif downsample == 'maxpool' and use_sn:
        downconv = [nn.MaxPool2d(2, stride=2),
                    nn.ReflectionPad2d(1),
                    spectral_norm(nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0))]

    else:
        raise NotImplementedError(
            'downsample layer [%s] 不存在' % downsample)
    return downconv


def upsampleLayer(inplanes, outplanes, upsample='basic', use_sn=True):
    # padding_type = 'zero'
    if upsample == 'basic' and not use_sn:
        upconv = [nn.ConvTranspose2d(
            inplanes, outplanes, kernel_size=3, stride=2, padding=1, output_padding=1)]
    elif upsample == 'bilinear' and not use_sn:
        upconv = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0)]
    elif upsample == 'nearest' and not use_sn:
        upconv = [nn.Upsample(scale_factor=2, mode='nearest'),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0)]
    elif upsample == 'subpixel' and not use_sn:
        upconv = [nn.Conv2d(inplanes, outplanes * 4, kernel_size=3, stride=1, padding=1),
                  nn.PixelShuffle(2)]
    elif upsample == 'basic' and use_sn:
        upconv = [spectral_norm(nn.ConvTranspose2d(
            inplanes, outplanes, kernel_size=3, stride=2, padding=1, output_padding=1))]
    elif upsample == 'bilinear' and use_sn:
        upconv = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                  nn.ReflectionPad2d(1),
                  spectral_norm(nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0))]
    elif upsample == 'nearest' and use_sn:
        upconv = [nn.Upsample(scale_factor=2, mode='nearest'),
                  nn.ReflectionPad2d(1),
                  spectral_norm(nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0))]
    elif upsample == 'subpixel' and use_sn:
        upconv = [spectral_norm(nn.Conv2d(inplanes, outplanes * 4, kernel_size=3, stride=1, padding=1)),
                  nn.PixelShuffle(2)]
    else:
        raise NotImplementedError(
            'upsample layer [%s] 不存在' % upsample)
    return upconv

class UnetGenerator(nn.Module):
    """使用Unet生成"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):

        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """
        U-net模块
        :param outer_nc:
        :param inner_nc:
        :param input_nc:
        :param submodule:
        :param outermost:
        :param innermost:
        :param norm_layer:
        :param use_dropout:
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


#####################################################################
#####   模型
#####################################################################
class VGGSimple(nn.Module):
    """Simonyan, K.; and Zisserman, A. 2014. https://arxiv.org/pdf/1409.1556.pdf"""

    def __init__(self):
        super(VGGSimple, self).__init__()

        self.features = self.make_layers()

        self.norm_mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.norm_std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, img, after_relu=True, base=8):
        """
        :param img:
        :param after_relu:
        :param base:
        :param base (int): img_size / 32
        :return:
        """
        # re-normalize from [-1, 1] to [0, 1] then to the range used for vgg
        feat = (((img + 1) * 0.5) - self.norm_mean.to(img.device)) / self.norm_std.to(img.device)
        # 提取特定VGG层的特征
        cut_points = [2, 7, 14, 21, 28]
        if after_relu:
            cut_points = [c + 2 for c in cut_points]
        for i in range(31):
            feat = self.features[i](feat)
            if i == cut_points[0]:
                feat_64 = F.adaptive_avg_pool2d(feat, base * 16)
            if i == cut_points[1]:
                feat_32 = F.adaptive_avg_pool2d(feat, base * 8)
            if i == cut_points[2]:
                feat_16 = F.adaptive_avg_pool2d(feat, base * 4)
            if i == cut_points[3]:
                feat_8 = F.adaptive_avg_pool2d(feat, base * 2)
            if i == cut_points[4]:
                feat_4 = F.adaptive_avg_pool2d(feat, base)

        return feat_64, feat_32, feat_16, feat_8, feat_4

    def make_layers(self, cfg="D", batch_norm=False):
        cfg_dic = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                  'M'],
        }
        cfg = cfg_dic[cfg]
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=False)]
                in_channels = v
        return nn.Sequential(*layers)


class SketchGenerator(nn.Module):
    def __init__(self, in_channel=512, hidden_channel=64, out_channel=3):
        """
        草图生成器
        :param in_channel (int): 输入图片的通道数
        :param hidden_channel (int): 中间隐藏层的通道数
        :param out_channel (int): 最终输出的通道数
        """
        super(SketchGenerator, self).__init__()

        self.decode_32 = UpConvBlock(in_channel, hidden_channel * 4)  # 32
        self.decode_64 = UpConvBlock(hidden_channel * 4, hidden_channel * 4)  # 64
        self.decode_128 = UpConvBlock(hidden_channel * 4, hidden_channel * 2)  # 128

        self.final = nn.Sequential(
            spectral_norm(nn.Conv2d(hidden_channel * 2, out_channel,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1),
                                    bias=True)),
            nn.Tanh())

    def forward(self, input):
        decode_32 = self.decode_32(input)
        decode_64 = self.decode_64(decode_32)
        decode_128 = self.decode_128(decode_64)

        output = self.final(decode_128)
        return output


class SketchDiscriminator(nn.Module):
    def __init__(self, hidden_channel=512, norm_layer='instance'):
        super(SketchDiscriminator, self).__init__()

        self.main = nn.Sequential(
            DownConvBlock(hidden_channel, hidden_channel // 2, norm_layer_type=norm_layer, down=False),
            DownConvBlock(hidden_channel // 2, hidden_channel // 4, norm_layer_type=norm_layer),  # 4x4
            spectral_norm(nn.Conv2d(hidden_channel // 4, 1, 4, 2, 0))
        )

    def forward(self, input):
        out = self.main(input)
        return out.view(-1)


class EncoderGenerator_Res(nn.Module):
    """
    把1 channel的edge图片加载为一个latent vector
    """

    def __init__(self, norm_layer='instance', image_size=256, input_nc=1, latent_dim=512):
        super(EncoderGenerator_Res, self).__init__()

        self.latent_dim = latent_dim
        latent_size = int(image_size / 32)
        self.latent_size = latent_size
        longsize = latent_dim * latent_size * latent_size
        self.longsize = longsize
        # print(image_size,latent_size, longsize)

        padding_type = 'reflect'

        layers_list = []
        # encode 256x256 -> 128x128
        layers_list.append(EncoderBlock(in_channel=input_nc, out_channel=32, kernel_size=4, padding=1, stride=2))

        dim_size = 32
        # dim:          32 -> 64 -> 128 -> 256 -> 512
        # image_size:   128 -> 64 -> 32 -> 16 -> 8
        for i in range(4):
            layers_list.append(ResBlock(dim_size, norm_type=norm_layer, padding_type=padding_type))
            layers_list.append(
                EncoderBlock(in_channel=dim_size, out_channel=dim_size * 2, kernel_size=4, padding=1, stride=2))
            dim_size *= 2

        layers_list.append(ResBlock(512, norm_type=norm_layer, padding_type=padding_type))
        self.conv = nn.Sequential(*layers_list)

        # final shape Bx512*8*8 -> Bx512
        self.fc_mu = nn.Sequential(nn.Linear(in_features=longsize, out_features=latent_dim))  # ,

        for m in self.modules():
            weights_init_normal(m)

    def forward(self, input):
        input = self.conv(input)
        ten = input.reshape(input.shape[0], -1)
        mu = self.fc_mu(ten)
        return mu


class DecoderGenerator_image_Res(nn.Module):
    """
    把latent vector 解码回 1 channel的灰度图
    """
    def __init__(self, norm_layer='instance', image_size=256, output_nc=1, latent_dim=512):
        super(DecoderGenerator_image_Res, self).__init__()

        self.latent_dim = latent_dim
        latent_size = int(image_size / 32)
        self.latent_size = latent_size
        # 512 * 8 * 8
        longsize = 512 * latent_size * latent_size

        padding_type = 'reflect'

        self.fc = nn.Sequential(nn.Linear(in_features=latent_dim, out_features=longsize))
        layers_list = []

        layers_list.append(ResBlock(512, norm_type=norm_layer, padding_type=padding_type))

        dim_size = 256
        # dim:          512 -> 256 -> 128 -> 64 -> 32
        # image size:   8 -> 16 -> 32 -> 64 -> 128
        for i in range(4):
            layers_list += [
                DecoderBlock(in_channel=dim_size * 2, out_channel=dim_size, kernel_size=4, padding=1, stride=2,
                             output_padding=0),
                ResBlock(dim_size, norm_type=norm_layer, padding_type=padding_type)]
            dim_size = int(dim_size / 2)

        # dim:  32 -> 21
        # size: 128 -> 256
        layers_list += [
            DecoderBlock(in_channel=32, out_channel=32, kernel_size=4, padding=1, stride=2, output_padding=0),
            ResBlock(32, norm_type=norm_layer, padding_type=padding_type)]

        layers_list.append(nn.ReflectionPad2d(2))
        layers_list.append(nn.Conv2d(32, output_nc, kernel_size=5, padding=0))

        self.conv = nn.Sequential(*layers_list)

        for m in self.modules():
            weights_init_normal(m)

    def forward(self, input):
        output = self.fc(input)
        out = output.reshape(output.shape[0], self.latent_dim, self.latent_size, self.latent_size)
        out = self.conv(out)
        return out


class DecoderGenerator_feature_Res(nn.Module):
    """
    输入: B x latent_dim
    输出: B x output_nc x image_size x image_size
    """

    def __init__(self, image_size, output_nc, latent_dim=512, norm_layer='instance'):
        super(DecoderGenerator_feature_Res, self).__init__()

        latent_size = int(image_size / 32)
        self.latent_size = latent_size
        longsize = 512 * latent_size * latent_size

        self.latent_dim = latent_dim

        padding_type = 'reflect'

        self.fc = nn.Sequential(nn.Linear(in_features=latent_dim, out_features=longsize))
        layers_list = []

        layers_list += [
            # 512x8x8 -> 512x8x8
            ResBlock(512, norm_type=norm_layer, padding_type=padding_type),
            # 512x8x8 -> 256x16x16
            DecoderBlock(in_channel=512, out_channel=256),
            # 256x16x16
            ResBlock(256, norm_type=norm_layer, padding_type=padding_type),
            # 256x16x16 -> 256x32x32
            DecoderBlock(in_channel=256, out_channel=256),
            # 256x32x32
            ResBlock(256, norm_type=norm_layer, padding_type=padding_type),
            # 128x64x64
            DecoderBlock(in_channel=256, out_channel=128),
            # 128x64x64
            ResBlock(128, norm_type=norm_layer, padding_type=padding_type),
            # 128x64x64 -> 64x128x128
            DecoderBlock(in_channel=128, out_channel=64),
            # 64x128x128
            ResBlock(64, norm_type=norm_layer, padding_type=padding_type),
            # 64x128x128 -> 64x256x256
            DecoderBlock(in_channel=64, out_channel=64),
            # 64x256x256
            ResBlock(64, norm_type=norm_layer, padding_type=padding_type),
            # 64x256x256->3x256x256
            nn.ReflectionPad2d(2),
            nn.Conv2d(in_channels=64, out_channels=output_nc, kernel_size=5, padding=0)
        ]

        self.conv = nn.Sequential(*layers_list)

        for m in self.modules():
            weights_init_normal(m)

    def forward(self, input):
        out = self.fc(input)
        out = out.reshape(out.shape[0], self.latent_dim, self.latent_size, self.latent_size)
        out = self.conv(out)
        return out


class ImageGenerator(nn.Module):
    """
    把feature map变为RGB image.
    输入 feature map : Bxf_CxHxW
    输出 RGB image : BxCxHxW
    """

    def __init__(self, input_nc, output_nc, hidden_nc=64, n_downsampling=3, n_blocks=9, norm_layer='batch',
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ImageGenerator, self).__init__()
        activation = nn.ReLU()

        # 3x256x256 -> 64x256x256
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, hidden_nc, kernel_size=7, padding=0),
                 get_norm_layer(hidden_nc, norm_type=norm_layer),
                 activation]

        ### downsample
        # 64x256x256 -> 128x128x128 -> 256x64x64 -> 512x32x32
        for i in range(n_downsampling):
            mult = (2 ** i)
            model += [nn.Conv2d((hidden_nc * mult), ((hidden_nc * mult) * 2), kernel_size=3, stride=2, padding=1),
                      get_norm_layer(((hidden_nc * mult) * 2), norm_type=norm_layer),
                      activation]

        ### resnet blocks
        # 512x32x32 -> 512x32x32
        mult = (2 ** n_downsampling)
        for i in range(n_blocks):
            model += [ResBlock((hidden_nc * mult), norm_type=norm_layer, padding_type=padding_type)]

        ### upsample
        # 512x32x32 -> 256x64x64 -> 128x128x128 -> 64x256x256
        for i in range(n_downsampling):
            mult = (2 ** (n_downsampling - i))
            model += [nn.ConvTranspose2d((hidden_nc * mult), int(((hidden_nc * mult) / 2)),
                                         kernel_size=3,
                                         stride=2,
                                         padding=1,
                                         output_padding=1),
                      get_norm_layer(int(((hidden_nc * mult) / 2)), norm_type=norm_layer)]

        # 64x256x256 -> 3x256x256
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(hidden_nc, output_nc, 7, padding=0),
                  nn.Tanh()]
        self.model = nn.Sequential(*model)

        for m in self.modules():
            weights_init_normal(m)

    def forward(self, input):
        return self.model(input)


class ImageDiscriminator(nn.Module):
    """
    输入3x256x256 -> 1x5x5
    使用PatchGAN的思想
    """

    def __init__(self, input_nc=3, ndf=64, image_size=256, norm_layer='instance'):
        super(ImageDiscriminator, self).__init__()
        self.input_nc = input_nc
        self.image_size = image_size

        padding_type = 'reflect'

        modules = [
            # 3x256x256 -> 16x128x128
            nn.Sequential(spectral_norm(nn.Conv2d(input_nc, ndf // 4, 4, 2, 1, bias=False)),
                          nn.LeakyReLU(0.2, inplace=True)),
            ResBlock(ndf // 4, norm_type=norm_layer, padding_type=padding_type),
            # 32x64x64
            DownConvBlock(ndf // 4, ndf // 2),
            ResBlock(ndf // 2, norm_type=norm_layer, padding_type=padding_type),
            # 64x32x32
            DownConvBlock(ndf // 2, ndf * 1),
            ResBlock(ndf * 1, norm_type=norm_layer, padding_type=padding_type),
            # 128x16x16
            DownConvBlock(ndf * 1, ndf * 2),
            ResBlock(ndf * 2, norm_type=norm_layer, padding_type=padding_type),
            # 256x8x8
            DownConvBlock(ndf * 2, ndf * 4),
            ResBlock(ndf * 4, norm_type=norm_layer, padding_type=padding_type),
            # 512x4x4
            nn.Sequential(
                spectral_norm(nn.Conv2d(ndf * 4, ndf * 4, 1, 1, 0, bias=False)),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False)))]

        self.main = nn.ModuleList(modules)
        for m in self.modules():
            weights_init_normal(m)

    def forward(self, input):
        # if input shape 256
        for layer in self.main:
            input = layer(input)
        return input


if __name__ == '__main__':
    # A = torch.randn(3, 3, 256, 256).cuda()
    # model = ImageDiscriminator().cuda()
    # print(model)
    # b = model(A)
    # print(b.shape)
    # print(b.mean())

    l = torch.tensor([1., 1., 1., 1.])
    model = WGANLoss()
    b = model(l, True)
    print(b)
    print(b.shape)
