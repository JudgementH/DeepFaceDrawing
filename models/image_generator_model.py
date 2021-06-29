import itertools
import os

import numpy as np
import torch
import torchsnooper
from torch import optim
import torch.nn.functional as F

from models import AutoEncoderModel
from models.base_model import BaseModel
from models.networks import DecoderGenerator_feature_Res, ImageGenerator, ImageDiscriminator, HingeLoss, VGGSimple, \
    compute_grad2, WGANLoss


class ImageGeneratorModel(BaseModel):
    def __init__(self, opt):
        super(ImageGeneratorModel, self).__init__(opt)
        self.image_size = opt.im_size
        self.latent_dim = opt.latent_dim

        # face part
        self.face_part = {'bg': (0, 0, 256),
                          'left_eye': (54, 78, 64),
                          'right_eye': (128, 78, 64),
                          'nose': (91, 116, 96),
                          'mouth': (84, 150, 96)}

        # 设置特征向量编码器
        self.autoencoder_weight_path = opt.autoencoder_weight_path
        self.__set_autoencoder()
        # 设置VGG
        if self.isTrain:
            self.vgg_weight_path = opt.vgg_weight_path
            self.__set_vgg()
        # feature list
        # self.bg_feature_list = None
        # self.left_eye_feature_list = None
        # self.right_eye_feature_list = None
        # self.nose_feature_list = None
        # self.mouth_feature_list = None
        self.feature_dict_list = {}

        # 5个特征解码器
        for key in self.face_part:
            part = self.face_part[key]
            net = DecoderGenerator_feature_Res(image_size=part[2],
                                               output_nc=opt.feature_dim,
                                               latent_dim=self.latent_dim,
                                               norm_layer=opt.norm)
            setattr(self, f'net_feature_decoder_{key}', net)
            decoder = getattr(self, f'net_feature_decoder_{key}')
            decoder.to(self.device)
            self.model_names += [f'net_feature_decoder_{key}']

        # 图片generator
        self.net_generator = ImageGenerator(input_nc=opt.feature_dim,
                                            output_nc=opt.output_dim,
                                            hidden_nc=opt.hidden_dim,
                                            n_blocks=opt.n_res_blocks,
                                            norm_layer=opt.norm)
        self.net_generator.to(self.device)

        if self.isTrain:
            # discriminator
            self.net_discriminator = ImageDiscriminator(input_nc=opt.output_dim,
                                                        image_size=self.image_size)
            self.net_discriminator.to(self.device)
            self.model_names += ['net_discriminator']

            # optimizer
            self.optimizer_g = optim.Adam(itertools.chain(self.net_feature_decoder_bg.parameters(),
                                                          self.net_feature_decoder_left_eye.parameters(),
                                                          self.net_feature_decoder_right_eye.parameters(),
                                                          self.net_feature_decoder_nose.parameters(),
                                                          self.net_feature_decoder_mouth.parameters(),
                                                          self.net_generator.parameters()),
                                          lr=opt.lr, betas=(0.5, 0.99))
            self.optimizer_d = optim.Adam(self.net_discriminator.parameters(),
                                          lr=opt.lr,
                                          betas=(0.5, 0.99))

            self.WGANLoss_func = WGANLoss()

    def name(self):
        return "ImageGenerator"

    def set_input(self, input):
        self.image = input['image'].to(self.device)
        self.edge = input['edge'].to(self.device)

    def set_edge(self, edge: torch.Tensor):
        self.edge = edge.to(self.device)

    def forward(self):
        self.latent_vector_dict = self.autoencoder.get_latent(self.edge)
        self.feature_map_bg = self.net_feature_decoder_bg(self.latent_vector_dict['bg_latent'])
        self.feature_map_left_eye = self.net_feature_decoder_left_eye(self.latent_vector_dict['left_eye_latent'])
        self.feature_map_right_eye = self.net_feature_decoder_right_eye(self.latent_vector_dict['right_eye_latent'])
        self.feature_map_nose = self.net_feature_decoder_nose(self.latent_vector_dict['nose_latent'])
        self.feature_map_mouth = self.net_feature_decoder_mouth(self.latent_vector_dict['mouth_latent'])

        self.feature_map = self.feature_map_bg.clone()
        for key in self.face_part:
            self.feature_map[:, :, self.face_part[key][0]:self.face_part[key][0] + self.face_part[key][2],
            self.face_part[key][1]:self.face_part[key][1] + self.face_part[key][2]] = getattr(self,
                                                                                              f'feature_map_{key}')
        self.fake_image = self.net_generator(self.feature_map)

    def generator_fake(self, latent_vector_dict: dict):
        """
        由dict中不同向量拆解为5个向量，再生成fake image
        :param latent_vector_dict(dict)
        """
        with torch.no_grad():
            self.feature_map = self.net_feature_decoder_bg(latent_vector_dict['bg_latent'])
            for key in latent_vector_dict:
                name = key.split("_latent")[0]
                net = getattr(self, f'net_feature_decoder_{name}')
                self.feature_map[:, :,
                self.face_part[name][0]:self.face_part[name][0] + self.face_part[name][2],
                self.face_part[name][1]:self.face_part[name][1] + self.face_part[name][
                    2]] = self.fake_image = net(latent_vector_dict[key])

            self.fake_image = self.net_generator(self.feature_map)

    def get_latent(self, edge: torch.Tensor):
        edge = edge.to(self.device)
        return self.autoencoder.get_latent(edge)

    def backward_d(self):

        # update loss with hinge loss
        # real sample
        # pred_real = self.net_discriminator(self.image)
        # hingeloss_real = self.HingeLoss_func(pred_real, True).mean()
        # hingeloss_real.backward()
        # self.loss_d_real = torch.sigmoid(pred_real.mean()).item()
        #
        # # fake sample
        # pred_fake = self.net_discriminator(self.fake_image.detach())
        # hingeloss_fake = self.HingeLoss_func(pred_fake, False).mean()
        # hingeloss_fake.backward()
        # self.loss_d_fake = torch.sigmoid(pred_fake.mean()).item()

        # update loss with wgangp
        # max real sample
        pred_real = self.net_discriminator(self.image)
        wganloss_real = self.WGANLoss_func(pred_real, True)

        # min
        pred_fake = self.net_discriminator(self.fake_image.detach())
        wganloss_fake = self.WGANLoss_func(pred_fake, False)

        self.WGANloss_d = -(wganloss_real + wganloss_fake) * 0.5
        self.WGANloss_d.backward()

        # wgan_gp
        self.reg = 10 * self.wgan_gp_reg(self.image, self.fake_image)
        self.reg.backward()

    def backward_g(self):
        pred = self.net_discriminator(self.fake_image)
        self.loss_g_gan = -pred.mean()
        self.loss_g_fake = torch.sigmoid(pred.mean()).item()

        self.loss_g_l1 = F.l1_loss(self.fake_image, self.image)

        fake_vgg_features = self.vgg(self.fake_image)
        real_vgg_features = self.vgg(self.image)

        self.loss_g_perceptual = 0.2 * (
                F.mse_loss(fake_vgg_features[3], real_vgg_features[3]) + \
                F.mse_loss(fake_vgg_features[2], real_vgg_features[2]) + \
                F.mse_loss(fake_vgg_features[1], real_vgg_features[1]))

        self.loss_g = self.loss_g_gan + self.opt.l1_weight * self.loss_g_l1 + self.opt.perceptual_weight * self.loss_g_perceptual
        # self.loss_g = self.loss_g_gan + self.opt.l1_weight * self.loss_g_l1
        # self.loss_g = self.opt.l1_weight * self.loss_g_l1
        self.loss_g.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_d.zero_grad()
        self.backward_d()
        self.optimizer_d.step()

        self.optimizer_g.zero_grad()
        self.backward_g()
        self.optimizer_g.step()

    def __set_autoencoder(self):
        # TODO autoencoder weight 的读取
        self.autoencoder = AutoEncoderModel(self.opt)
        self.autoencoder.load_networks(0)
        self.autoencoder.eval()
        for p in self.autoencoder.net_encoder_bg.parameters():
            p.requires_grad = False
        for p in self.autoencoder.net_encoder_left_eye.parameters():
            p.requires_grad = False
        for p in self.autoencoder.net_encoder_right_eye.parameters():
            p.requires_grad = False
        for p in self.autoencoder.net_encoder_nose.parameters():
            p.requires_grad = False
        for p in self.autoencoder.net_encoder_mouth.parameters():
            p.requires_grad = False
        if self.isTrain:
            for p in self.autoencoder.net_decoder_bg.parameters():
                p.requires_grad = False
            for p in self.autoencoder.net_decoder_left_eye.parameters():
                p.requires_grad = False
            for p in self.autoencoder.net_decoder_right_eye.parameters():
                p.requires_grad = False
            for p in self.autoencoder.net_decoder_nose.parameters():
                p.requires_grad = False
            for p in self.autoencoder.net_decoder_mouth.parameters():
                p.requires_grad = False

    def __set_vgg(self):
        self.vgg = VGGSimple()
        self.vgg.load_state_dict(torch.load(self.vgg_weight_path, map_location=lambda a, b: a))
        self.vgg.to(self.device)
        self.vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

    def wgan_gp_reg(self, x_real, x_fake, center=1.):
        batch_size = x_real.shape[0]
        eps = torch.rand(batch_size, device=x_real.device).view(batch_size, 1, 1, 1)
        x_interp = (1 - eps) * x_real + eps * x_fake
        x_interp = x_interp.detach()
        x_interp.requires_grad_()
        d_out = self.net_discriminator(x_interp)
        reg = (compute_grad2(d_out, x_interp).sqrt() - center).pow(2).mean()
        return reg

    def add_feature_list(self, edge: torch.Tensor):
        # 把edge提取出latent，并加入feature list:numpy
        edge = edge.to(self.device)
        feature_list_dict = self.autoencoder.get_latent(edge)  # Bx512
        if not self.feature_dict_list:
            self.feature_dict_list = feature_list_dict
        else:
            # self.feature_dict_list = torch.cat((self.feature_list, feature_list), dim=0)
            for key in feature_list_dict:
                self.feature_dict_list[key] = torch.cat((self.feature_dict_list[key], feature_list_dict[key]), dim=0)

    def save_feature_list(self, save_name: str):
        # 保存地址 self.save_path/save_name ,save_name with npy
        save_path = os.path.join(self.save_dir, save_name)
        feature_dict_np = {}
        for key in self.feature_dict_list:
            feature_dict_np[key] = self.feature_dict_list[key].cpu().numpy()
        np.save(save_path, feature_dict_np)

    def load_feature_list(self, file_name: str):
        # load self.save_path/file_name 文件
        load_filename = os.path.join(self.save_dir, file_name)
        feature_list_np = np.load(load_filename, allow_pickle=True).item()
        for key in feature_list_np:
            self.feature_dict_list[key] = torch.from_numpy(feature_list_np[key]).to(self.device)

    def get_latent_dict(self, input_latent_dict: dict, nearN=3, gate=0.) -> dict:
        """
        从self.feature_dict_list中找相似的latent
        :param input_latent_dict (dict(torch.Tensor)): 一个dict内部有5个tensor
                                                      key值为bg_latent，left_eye_latent，right_eye_latent，nose_latent，mouth_latent
        :param nearN (int): 近邻数量
        :param gate (float): 插值在近邻和输入之间插值，
        :return similar_latent_dict (dict(torch.Tensor)): 5个相似向量组成的dict
        """
        similar_latent_dict = {}
        for key in input_latent_dict:
            input_latent = input_latent_dict[key]
            similar_latent_dict[key] = self.get_similar_latent_part(input_latent, key, nearN, gate)
        return similar_latent_dict

    def get_similar_latent_part(self, input_latent: torch.Tensor, part: str, nearN=3, gate=0.) -> torch.Tensor:
        """
        按照不同的part找相似latent_vector
        :param input_latent (torch.Tensor): input_latent should be in shape 1 x latent_dim
        :param part (str): select a part bg_latent|left_eye_latent|right_eye_latent|nose_latent|mouth_latent
        :param nearN (int): 近邻数量
        :param gate (float): 插值在近邻和输入之间插值，
        :return latent_inter (torch.Tensor): latent_inter = (1-gate) * near_latent + gate* input_latent
        """
        key = part
        feature_list = self.feature_dict_list[key]
        euclid_dist = torch.norm(self.feature_dict_list[key] - input_latent, dim=1)
        dis_min_k, idx_min_k = torch.topk(euclid_dist, nearN, largest=False)
        similar_k_latent = torch.index_select(feature_list, dim=0, index=idx_min_k)
        A = similar_k_latent.T
        A = torch.cat((A, torch.ones(1, nearN).to(self.device)), dim=0)
        b = input_latent.T
        b = torch.cat((b, torch.zeros(1, 1).to(self.device)), dim=0)
        x, _ = torch.lstsq(b, A)
        x = x[:nearN]

        near_latent = x.T @ similar_k_latent

        latent_inter = (1 - gate) * near_latent + gate * input_latent
        return latent_inter


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)
    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product
    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


def gram_loss(input, target):
    in_gram = gram_matrix(input)
    tar_gram = gram_matrix(target.detach())
    return F.mse_loss(in_gram, tar_gram)
