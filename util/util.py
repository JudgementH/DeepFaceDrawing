import os

import numpy as np
import torch
from PIL import Image


def save_image(image_tensor, save_filename):
    """
    把tensor类型的image保存在save_path目录下
    :param image_tensor (tensor):要保存的图片
    :param save_filename (str): 要保存的地址，包含图片文件名称如 images/1.jpg
    """
    image_np = tensor2im(image_tensor)
    image_pil = Image.fromarray(image_np)
    dirname = os.path.dirname(save_filename)
    if not os.path.exists(dirname):
        mkdir(dirname)
    image_pil.save(save_filename)


def tensor2im(input_image, imtype=np.uint8):
    """把tensor变为一个numpy的arrary
    Parameters:
        input_image (tensor) --  tensor类型的图片
        imtype (type)        --  目标numpy数据类型
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def mkdirs(paths):
    """创建文件夹列表
    Parameters:
        paths (list[str]) -- 一个文件夹路径的list
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """若不存在目标文件夹，会创建文件夹及其父目录
    Parameters:
        path (str) -- 文件夹路径
    """
    if not os.path.exists(path):
        os.makedirs(path)
