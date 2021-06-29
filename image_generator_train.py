import torch
from tqdm import tqdm

import data
import models

import torchvision.utils as vutils

from options.model_options import ImageGeneratorOption

if __name__ == '__main__':
    # train dataset
    train_auto_encoder_parser = ImageGeneratorOption()
    train_auto_encoder_parser.initialize()
    train_auto_encoder_parser.set_train_option()
    trian_opt = train_auto_encoder_parser.parse()
    train_dataloader = data.create_dataloader(trian_opt)

    # test dataset
    test_auto_encoder_parser = ImageGeneratorOption()
    test_auto_encoder_parser.initialize()
    test_auto_encoder_parser.set_test_option()
    test_opt = test_auto_encoder_parser.parse()
    test_dataloader = data.create_dataloader(test_opt)

    model = models.create_model(trian_opt)
    min_loss = 1e10
    which_epoch = -1
    for epoch in range(trian_opt.epoch):
        # train
        train_dataloader = tqdm(train_dataloader)
        train_dataloader.set_description(
            '[%s%03d/%03d %s=%f]' % ('Epoch:', epoch + 1, trian_opt.epoch, 'lr', trian_opt.lr))

        for i, data in enumerate(train_dataloader):
            model.set_input(data)
            model.optimize_parameters()
            train_dataloader.set_postfix({'loss_d': model.WGANloss_d.item(),
                                          'loss_g': model.loss_g.item()})

            if i % 100 == 0:
                # save train fake image
                si = torch.cat([model.image, torch.cat([model.edge, model.edge, model.edge], dim=1), model.fake_image],
                               dim=0)
                vutils.save_image(si, 'datasets/image_generator/epoch_%d_i_%d.jpg' % (epoch, i), range=(-1, 1),
                                  normalize=True)
                if i > 0:
                    break

        # test_loss = 0
        test_dataloader = tqdm(test_dataloader)
        for i, data in enumerate(test_dataloader):
            model.set_input(data)
            model.test()
            si = torch.cat([model.image, torch.cat([model.edge, model.edge, model.edge], dim=1), model.fake_image],
                           dim=0)
            vutils.save_image(si, 'datasets/image_generator_test/epoch_%d_i_%d.jpg' % (epoch, i), range=(-1, 1),
                              normalize=True)

        # save the model
        model.save_networks(0)
