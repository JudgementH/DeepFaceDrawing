import os

from PIL import Image

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import ImageFolder


class ImageDataset(BaseDataset):

    def __init__(self, opt):
        super(ImageDataset, self).__init__(opt)
        self.image_paths = ImageFolder(os.path.join(self.root, opt.path_image))

        gray = False
        if hasattr(opt,'gray'):
            gray = opt.gray

        hFlip = False
        if hasattr(opt,'h_flip'):
            hFlip = opt.h_flip

        self.transform = get_transform(grayscale=gray, hFlip=hFlip)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image_name = os.path.basename(self.image_paths[item])
        image_pil = Image.open(self.image_paths[item]).convert("RGB")
        image_tensor = self.transform(image_pil)
        return {"image": image_tensor, "name": image_name}
