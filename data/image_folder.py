import torch.utils.data as data
from PIL import Image
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


class ImageFolder(data.Dataset):
    """保存image的类"""
    def __init__(self, root):
        """
        :param root: 图片文件夹地址，内部应该是多个图片
        """
        super(ImageFolder, self).__init__()
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise (RuntimeError("目录下没有图片，目录: " + root
                                + "\n目前支持的图片格式有: " + ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        """
        取第item个图片
        :param item (int): 序号数
        :return path (str): 第item个图片的地址
        """
        path = self.imgs[item]
        return path


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    """
    :param dir (str): 图片文件夹，内部应是很多图片文件
    :param max_dataset_size (float): dataset的最大数目
    :return: images (list[str]): 返回一个list，list内部是每一个图片的路径，顺序按照字典序排列
    """
    images = []
    assert os.path.isdir(dir), '%s 不是文件夹' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]
