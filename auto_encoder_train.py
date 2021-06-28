import torch
from tqdm import tqdm

import data
import models
from options.model_options import AutoEncoderOption

import torchvision.utils as vutils

if __name__ == '__main__':
    # train dataset
    train_auto_encoder_parser = AutoEncoderOption()
    train_auto_encoder_parser.initialize()
    train_auto_encoder_parser.set_train_option()
    ae_opt = train_auto_encoder_parser.parse()
    train_dataloader = data.create_dataloader(ae_opt)

    # test dataset
    test_auto_encoder_parser = AutoEncoderOption()
    test_auto_encoder_parser.initialize()
    test_auto_encoder_parser.set_test_option()
    test_opt = test_auto_encoder_parser.parse()
    test_dataloader = data.create_dataloader(test_opt)

    model = models.create_model(ae_opt)
    min_loss = 1e10
    which_epoch = -1
    for epoch in range(ae_opt.epoch):
        # train
        train_dataloader = tqdm(train_dataloader)
        train_dataloader.set_description(
            '[%s%03d/%03d %s=%f]' % ('Epoch:', epoch + 1, ae_opt.epoch, 'lr', ae_opt.lr))
        loss = 0
        for i, data in enumerate(train_dataloader):
            model.set_input(data)
            model.optimize_parameters()
            loss += model.loss.cpu().mean().item()
            train_dataloader.set_postfix({'loss': loss / (i + 1)})

            if i % 10 == 0:
                # save train fake image
                si = torch.cat([model.image, model.bg_fake], dim=0)
                vutils.save_image(si, 'datasets/autoencoder/epoch_%d_i_%d.jpg' % (epoch, i), range=(-1, 1),
                                  normalize=True)


        test_loss = 0
        test_dataloader = tqdm(test_dataloader)
        for i, data in enumerate(test_dataloader):
            model.set_test_input(data)
            model.test()
            test_loss += model.loss.cpu().mean().item()
            si = torch.cat([model.image, model.fake], dim=0)
            test_dataloader.set_postfix({'test_loss': test_loss / (i + 1)})
            vutils.save_image(si, 'datasets/autoencoder_test/epoch_%d_i_%d.jpg' % (epoch, i), range=(-1, 1),
                              normalize=True)

        # save the model
        if test_loss < min_loss:
            min_loss = test_loss
            which_epoch = epoch
            model.save_networks(0)

    print(f"save the best model with test loss: {min_loss} in epoch:{which_epoch}")
