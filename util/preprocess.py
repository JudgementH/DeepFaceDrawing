import os

from PIL import Image
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def makedataset(dir):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


def pad_image(image):
    w, h = image.size
    target = max(w, h)
    x = 0 if w == target else (target - w) // 2
    y = 0 if h == target else (target - h) // 2
    new_image = Image.new('RGB', (target, target), (255, 255, 255))
    new_image.paste(image, (x, y))
    return new_image


def resize_and_save(image_dir, output_dir, target_size=256):
    """
    把image_dir内的图片转化为256*256大小的图片
    :param image_dir (str): 图片文件夹地址，内部有很多图片
    :param output_dir (str): 输出目标文件夹地址，会自动创建
    :param target_size (int): 输出目标的大小
    :return:
    """
    images_path = makedataset(image_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i in range(len(images_path)):
        image = Image.open(images_path[i])
        image = pad_image(image)
        image = image.resize((target_size, target_size), Image.BICUBIC)
        image.save(os.path.join(output_dir, os.path.basename(images_path[i])))
        print('%d/%d finished!' % (i, len(images_path)))


def main():
    path = './mydataset/train/artist/car/00004.png'
    image = Image.open(path)
    print(len(image.split()))


if __name__ == '__main__':
    DIR = r'F:\DD\datasets\Face\face'
    OUT = r'F:\DD\datasets\Face\face512'
    # resize_and_save(DIR, OUT, target_size=512)

    # a_np = np.array([1, 2, 3])
    # b_np = np.array([4, 5, 6])
    # dict_ = {'a': a_np, 'b': b_np}
    # np.save('data.npy',dict_)
    # dict_ = np.load('data.npy',allow_pickle=True).item()
    # print(type(dict_))
    # for key in dict_:
    #     print(key)

