from abc import ABC, abstractmethod
from PIL import Image

import torch.utils.data as data
import torchvision.transforms as transforms


class BaseDataset(data.Dataset):
    """Dataset抽象类
       需要实现以下函数
       -- <__init__>:                      构造函数
       -- <__len__>:                       dataset的长度
       -- <__getitem__>:                   获取一个数据对象
    """

    def __init__(self, opt):
        """构造函数，保存option

        :param  opt (Option class): stores all the experiment options; needs to be a subclass of BaseOptions
        """
        super(BaseDataset, self).__init__()
        self.opt = opt
        self.root = opt.dataroot

    def __len__(self):
        """返回数据集长度"""
        return 0

    def __getitem__(self, item):
        """从数据集获取第item个数据
                    :param  index (int): 目标数据的index号
                    :return data (dict): dict 包含了数据和其他信息
                    """
        pass


def get_transform(grayscale=False, convert=True, hFlip=False):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    if hFlip:
        transform_list += [transforms.RandomHorizontalFlip()]

    if convert:
        transform_list += [transforms.ToTensor()]

        # turn [0,1]  to [-1,1]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def _rescale(img):
    return img * 2.0 - 1.0


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_size, crop_size, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """warning information关于图片大小(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("图片大小需要是4的倍数。"
              "目前加载的图片大小是 (%d, %d), 我们将其调整为(%d, %d)。"
              "我们会对所有图片大小"
              "不是4的倍数的图片进行变换" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
