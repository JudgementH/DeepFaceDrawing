from options.base_options import BaseOption

class AutoEncoderOption(BaseOption):
    def initialize(self):
        BaseOption.initialize(self)
        self.parser.set_defaults(name='auto_encoder_option')
        self.parser.add_argument('--model', type=str, default='autoencoder', help='决定选用的model')

        # 文件地址
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/auto_encoder/',
                                 help='模型保存根地址')
        self.parser.add_argument('--dataroot', type=str, default='./datasets/', help='数据集根目录地址')

        # 模型参数
        self.parser.add_argument('--im_size', type=int, default=256, help='resolution of the generated images')
        self.parser.add_argument('--latent_dim', type=int, default=512, help='encoder后向量的维数')
        self.parser.add_argument('--input_nc', type=int, default=1, help='输入图片的维数')
        self.parser.add_argument('--mse_weight', default=0.2, type=float,
                                 help='let G generate images with content more like in set input image')

        # 数据集参数
        self.parser.add_argument('--gray', type=bool, default=True, help='输入控制为灰度图')

    def set_train_option(self):
        # 决定使用的dataset
        self.parser.set_defaults(name='auto_encoder_train_option')
        self.parser.add_argument('--isTrain', type=bool, default=True)

        # 文件地址
        self.parser.add_argument('--path_image', type=str, default='train/Edge', help='边缘图片文件夹地址，内部应该有很多图片')

        # 训练参数
        self.parser.add_argument('--epoch', type=int, default=200, help='训练的epoch数目')
        self.parser.add_argument('--lr', type=float, default=2e-4,
                                 help='learning rate, default is 2e-4, usually dont need to change it, you can try make it smaller, such as 1e-4')

        # 数据集参数
        self.parser.add_argument('--batch_size', type=int, default=1,
                                 help='how many images to train together at one iteration')
        self.parser.add_argument('--shuffle', type=bool, default=True, help='数据集是否打乱')

    def set_test_option(self):
        self.parser.set_defaults(name='auto_encoder_test_option')
        self.parser.add_argument('--isTrain', type=bool, default=False)

        # 文件地址
        self.parser.add_argument('--path_image', type=str, default='test/Edge', help='边缘图片文件夹地址，内部应该有很多图片')

        # 数据集参数
        self.parser.add_argument('--batch_size', type=int, default=1,
                                 help='how many images to train together at one iteration')
        self.parser.add_argument('--shuffle', type=bool, default=False, help='数据集是否打乱')


class ImageGeneratorOption(BaseOption):
    def initialize(self):
        BaseOption.initialize(self)
        self.parser.set_defaults(name='image_generator_option')
        self.parser.add_argument('--model', type=str, default='image_generator', help='决定选用的model')

        # 文件地址
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/image_generator/',
                                 help='模型保存根地址')
        self.parser.add_argument('--dataroot', type=str, default='./datasets/', help='数据集根目录地址')
        # TODO autoencoder path
        self.parser.add_argument('--autoencoder_weight_path', type=str,
                                 default='./checkpoints/auto_encoder/AutoEncoder0.pth',
                                 help='autoencoder网络权值地址')

        # 模型参数
        self.parser.add_argument('--im_size', type=int, default=256, help='resolution of the generated images')
        self.parser.add_argument('--latent_dim', type=int, default=512, help='feature vector长度')
        self.parser.add_argument('--feature_dim', type=int, default=32, help='feature map 通道数')
        self.parser.add_argument('--input_nc', type=int, default=1, help='sketch channel')
        self.parser.add_argument('--output_dim', type=int, default=3, help='output image dim')
        self.parser.add_argument('--hidden_dim', type=int, default=56, help='hidden dim in conv layer')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance | batch')
        self.parser.add_argument('--n_downsmalpe', type=int, default=4, help='conv中下采样的层数')
        self.parser.add_argument('--n_res_blocks', type=int, default=9, help='generator中resblock层数')

        # 数据集参数
        self.parser.add_argument('--gray', type=bool, default=True, help='输入控制为灰度图')

    def set_train_option(self):
        # TODO 修改train、test方法
        # 决定使用的dataset
        self.parser.set_defaults(name='image_generator_train_option')
        self.parser.add_argument('--isTrain', type=bool, default=True)

        # 文件地址
        self.parser.add_argument('--vgg_weight_path', type=str, default='./checkpoints/VGG/vgg-feature-weights.pth',
                                 help='VGG网络权值地址')
        self.parser.add_argument('--path_image', type=str, default='train/Image', help='彩色图片文件夹地址，内部应该有很多图片')
        self.parser.add_argument('--path_edge', type=str, default='train/Edge', help='边缘图片文件夹地址，内部应该有很多图片')

        # 训练参数
        self.parser.add_argument('--epoch', type=int, default=200, help='训练的epoch数目')
        self.parser.add_argument('--lr', type=float, default=2e-4,
                                 help='learning rate, default is 2e-4, usually dont need to change it, you can try make it smaller, such as 1e-4')
        self.parser.add_argument('--l1_weight', type=float, default=1, help='l1 loss weight')
        self.parser.add_argument('--perceptual_weight', type=float, default=1, help='perceptual loss weight')

        # 数据集参数
        self.parser.add_argument('--batch_size', type=int, default=1,
                                 help='how many images to train together at one iteration')
        self.parser.add_argument('--shuffle', type=bool, default=True, help='数据集是否打乱')

    def set_test_option(self):
        self.parser.set_defaults(name='image_generator_test_option')
        self.parser.add_argument('--isTrain', type=bool, default=False)

        # 文件地址
        self.parser.add_argument('--path_image', type=str, default='test/Image', help='彩色图片文件夹地址，内部应该有很多图片')
        self.parser.add_argument('--path_edge', type=str, default='test/Edge', help='edge图片文件夹地址，内部应该有很多图片')

        # 数据集参数
        self.parser.add_argument('--batch_size', type=int, default=1,
                                 help='how many images to train together at one iteration')
        self.parser.add_argument('--shuffle', type=bool, default=False, help='数据集是否打乱')

    def set_execution_option(self):
        self.parser.set_defaults(name='image_generator_execution_option')
        self.parser.add_argument('--isTrain', type=bool, default=False)
