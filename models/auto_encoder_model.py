import itertools

from torch import optim
import torch.nn.functional as F

from models.base_model import BaseModel
from models.networks import EncoderGenerator_Res, DecoderGenerator_image_Res


class AutoEncoderModel(BaseModel):
    """
    AutoEncoder
    this model will encode a edge to a vector
    """

    def __init__(self, opt):
        super(AutoEncoderModel, self).__init__(opt)

        self.image_size = opt.im_size
        self.input_nc = opt.input_nc

        self.latent_dim = opt.latent_dim

        # face part
        self.face_part = {'bg': (0, 0, 256),
                          'left_eye': (54, 78, 64),
                          'right_eye': (128, 78, 64),
                          'nose': (91, 116, 96),
                          'mouth': (84, 150, 96)}

        # 5 encoders
        self.net_encoder_bg = EncoderGenerator_Res(image_size=self.face_part['bg'][2],
                                                   input_nc=self.input_nc,
                                                   latent_dim=self.latent_dim)
        self.net_encoder_bg.to(self.device)
        self.model_names += ['net_encoder_bg']

        self.net_encoder_left_eye = EncoderGenerator_Res(image_size=self.face_part['left_eye'][2],
                                                         input_nc=self.input_nc,
                                                         latent_dim=self.latent_dim)
        self.net_encoder_left_eye.to(self.device)
        self.model_names += ['net_encoder_left_eye']

        self.net_encoder_right_eye = EncoderGenerator_Res(image_size=self.face_part['right_eye'][2],
                                                          input_nc=self.input_nc,
                                                          latent_dim=self.latent_dim)
        self.net_encoder_right_eye.to(self.device)
        self.model_names += ['net_encoder_right_eye']

        self.net_encoder_nose = EncoderGenerator_Res(image_size=self.face_part['nose'][2],
                                                     input_nc=self.input_nc,
                                                     latent_dim=self.latent_dim)
        self.net_encoder_nose.to(self.device)
        self.model_names += ['net_encoder_nose']

        self.net_encoder_mouth = EncoderGenerator_Res(image_size=self.face_part['mouth'][2],
                                                      input_nc=self.input_nc,
                                                      latent_dim=self.latent_dim)
        self.net_encoder_mouth.to(self.device)
        self.model_names += ['net_encoder_mouth']

        # decoder
        if opt.isTrain:
            self.net_decoder_bg = DecoderGenerator_image_Res(image_size=self.face_part['bg'][2],
                                                             output_nc=self.input_nc,
                                                             latent_dim=self.latent_dim)
            self.net_decoder_bg.to(self.device)
            self.model_names += ['net_decoder_bg']

            self.net_decoder_left_eye = DecoderGenerator_image_Res(image_size=self.face_part['left_eye'][2],
                                                                   output_nc=self.input_nc,
                                                                   latent_dim=self.latent_dim)
            self.net_decoder_left_eye.to(self.device)
            self.model_names += ['net_decoder_left_eye']

            self.net_decoder_right_eye = DecoderGenerator_image_Res(image_size=self.face_part['right_eye'][2],
                                                                    output_nc=self.input_nc,
                                                                    latent_dim=self.latent_dim)
            self.net_decoder_right_eye.to(self.device)
            self.model_names += ['net_decoder_right_eye']

            self.net_decoder_nose = DecoderGenerator_image_Res(image_size=self.face_part['nose'][2],
                                                               output_nc=self.input_nc,
                                                               latent_dim=self.latent_dim)
            self.net_decoder_nose.to(self.device)
            self.model_names += ['net_decoder_nose']

            self.net_decoder_mouth = DecoderGenerator_image_Res(image_size=self.face_part['mouth'][2],
                                                                output_nc=self.input_nc,
                                                                latent_dim=self.latent_dim)
            self.net_decoder_mouth.to(self.device)
            self.model_names += ['net_decoder_mouth']

            self.optimizer = optim.Adam(itertools.chain(self.net_encoder_bg.parameters(),
                                                        self.net_encoder_left_eye.parameters(),
                                                        self.net_encoder_right_eye.parameters(),
                                                        self.net_encoder_nose.parameters(),
                                                        self.net_encoder_mouth.parameters(),
                                                        self.net_decoder_bg.parameters(),
                                                        self.net_decoder_left_eye.parameters(),
                                                        self.net_decoder_right_eye.parameters(),
                                                        self.net_decoder_nose.parameters(),
                                                        self.net_decoder_mouth.parameters()),
                                        lr=opt.lr, betas=(0.5, 0.999))

    def name(self):
        return "AutoEncoder"

    def set_input(self, input):
        self.image = input['image'].to(self.device)
        leye_attr = self.face_part['left_eye']
        reye_attr = self.face_part['right_eye']
        nose_attr = self.face_part['nose']
        mouth = self.face_part['mouth']
        self.left_eye = self.image[:, :, leye_attr[0]:leye_attr[0] + leye_attr[2],
                        leye_attr[1]:leye_attr[1] + leye_attr[2]]
        self.right_eye = self.image[:, :, reye_attr[0]:reye_attr[0] + reye_attr[2],
                         reye_attr[1]:reye_attr[1] + reye_attr[2]]
        self.nose = self.image[:, :, nose_attr[0]:nose_attr[0] + nose_attr[2],
                    nose_attr[1]:nose_attr[1] + nose_attr[2]]
        self.mouth = self.image[:, :, mouth[0]:mouth[0] + mouth[2],
                     mouth[1]:mouth[1] + mouth[2]]
        bg_loc = self.image.clone()
        for key in self.face_part:
            part = self.face_part[key]
            bg_loc[:, :, part[0]:part[0] + part[2], part[1]:part[1] + part[2]] = 1
        self.bg = bg_loc

    def forward(self):
        # self.latent_vector = self.net_encoder(self.image)
        # self.fake = self.net_decoder(self.latent_vector)
        # get latent
        self.bg_latent = self.net_encoder_bg(self.bg)
        self.left_eye_latent = self.net_encoder_left_eye(self.left_eye)
        self.right_eye_latent = self.net_encoder_right_eye(self.right_eye)
        self.nose_latent = self.net_encoder_nose(self.nose)
        self.mouth_latent = self.net_encoder_mouth(self.mouth)

        # decode
        self.bg_fake = self.net_decoder_bg(self.bg_latent)
        self.left_eye_fake = self.net_decoder_left_eye(self.left_eye_latent)
        self.right_eye_fake = self.net_decoder_right_eye(self.right_eye_latent)
        self.nose_fake = self.net_decoder_nose(self.nose_latent)
        self.mouth_fake = self.net_decoder_mouth(self.mouth_latent)

        self.fake = self.bg_fake.clone()
        for key in self.face_part:
            self.fake[:, :, self.face_part[key][0]:self.face_part[key][0] + self.face_part[key][2],
            self.face_part[key][1]:self.face_part[key][1] + self.face_part[key][2]] = getattr(self, f'{key}_fake')

    def backward(self):
        bg_mse = F.mse_loss(self.bg, self.bg_fake)
        left_eye_mse = F.mse_loss(self.left_eye, self.left_eye_fake)
        right_eye_mse = F.mse_loss(self.right_eye, self.right_eye_fake)
        nose_mse = F.mse_loss(self.nose, self.nose_fake)
        mouth_mse = F.mse_loss(self.mouth, self.mouth_fake)
        self.loss_mse = bg_mse + left_eye_mse + right_eye_mse + nose_mse + mouth_mse
        self.loss = self.opt.mse_weight * self.loss_mse
        self.loss.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def set_test_input(self, input):
        self.image = input['image'].to(self.device)


    def get_latent(self, image):
        self.image = image.to(self.device)
        leye_attr = self.face_part['left_eye']
        reye_attr = self.face_part['right_eye']
        nose_attr = self.face_part['nose']
        mouth = self.face_part['mouth']
        self.left_eye = self.image[:, :, leye_attr[0]:leye_attr[0] + leye_attr[2],
                        leye_attr[1]:leye_attr[1] + leye_attr[2]]
        self.right_eye = self.image[:, :, reye_attr[0]:reye_attr[0] + reye_attr[2],
                         reye_attr[1]:reye_attr[1] + reye_attr[2]]
        self.nose = self.image[:, :, nose_attr[0]:nose_attr[0] + nose_attr[2],
                    nose_attr[1]:nose_attr[1] + nose_attr[2]]
        self.mouth = self.image[:, :, mouth[0]:mouth[0] + mouth[2],
                     mouth[1]:mouth[1] + mouth[2]]
        bg_loc = self.image.clone()
        for key in self.face_part:
            part = self.face_part[key]
            bg_loc[:, :, part[0]:part[0] + part[2], part[1]:part[1] + part[2]] = 1
        self.bg = bg_loc

        return {'bg_latent': self.net_encoder_bg(self.bg),
                'left_eye_latent': self.net_encoder_left_eye(self.left_eye),
                'right_eye_latent': self.net_encoder_right_eye(self.right_eye),
                'nose_latent': self.net_encoder_nose(self.nose),
                'mouth_latent': self.net_encoder_mouth(self.mouth)}
