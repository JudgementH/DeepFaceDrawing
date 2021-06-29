import os
import numpy as np
from PIL import Image
from torchvision import transforms

import data
from data.image_folder import make_dataset
import models
from options.model_options import ImageGeneratorOption
import torchvision.utils as vutils

image_generator_parser = ImageGeneratorOption()
image_generator_parser.initialize()
image_generator_parser.set_execution_option()
opt = image_generator_parser.parse()

model = models.create_model(opt)
model.load_networks(0)
model.load_feature_list("feature.npy")


def get_transform(grayscale=True, convert=True):
    transform_list = [transforms.Resize((256, 256))]
    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    if convert:
        transform_list += [transforms.ToTensor()]

        # turn [0,1]  to [-1,1]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def process_edge(edge_src):
    transform = get_transform()

    images = make_dataset(edge_src)
    length = len(images)
    for i, filenmae in enumerate(images):
        print("processing image %d/%d" % ((i + 1), length), end='\t')
        edge_pil = Image.open(filenmae).convert("RGB")
        edge_tensor = transform(edge_pil).unsqueeze(0)
        model.add_feature_list(edge_tensor)
        print("finished!!")
    model.save_feature_list("feature.npy")
    print("-" * 80)
    print("All finished!!!")


def execute_model(image_path: str, save_name, nearN=2):
    edge_pil = Image.open(image_path).convert("RGB")

    transform = get_transform()
    edge_tensor = transform(edge_pil)
    edge_tensor.unsqueeze_(0)  # edge_tensor 大小应为[1,1,256,256]
    edge_latent_tensor = model.get_latent(edge_tensor)
    similar_latent = model.get_latent_dict(edge_latent_tensor, nearN=nearN)
    model.generator_fake(similar_latent)
    # model.set_edge(edge_tensor)
    # model.test()
    vutils.save_image(model.fake_image, '%s' % (save_name), range=(-1, 1), normalize=True)


if __name__ == '__main__':
    src = r"./datasets/test/Edge"
    # process_edge(src)

    sketch_src = r"./datasets/test/Edge/13442.jpg"
    i = "1.jpg``````````````````````````````````````"
    execute_model(sketch_src, i)

    # test dataset
    # images = make_dataset(r"E:\a")
    #
    # for i, image in enumerate(images):
    #     print(i)
    #     name = os.path.basename(image).split('.')[0]
    #     execute_model(image, f'E:/a/a_{name}.jpg', nearN=10)
