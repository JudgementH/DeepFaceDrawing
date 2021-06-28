from models.auto_encoder_model import AutoEncoderModel
from models.image_generator_model import ImageGeneratorModel


def create_model(opt):
    model_name = opt.model
    if model_name == 'autoencoder':
        model = AutoEncoderModel(opt)
    elif model_name == 'image_generator':
        model = ImageGeneratorModel(opt)
    else:
        raise ValueError("不存在option.model [%s] 所对应的数据集" % model_name)
    return model
