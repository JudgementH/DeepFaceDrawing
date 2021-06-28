from torch.utils.data import DataLoader


def create_dataloader(opt):
    """
    从opt参数中创建dataloader
    :param opt (<Option>): a subclass of BaseOption
    :return dataloader (<DataLoader>): A class of torch.utils.data.Dataloader
    """
    name = opt.name
    if name == 'auto_encoder_train_option' or name == 'auto_encoder_test_option':
        from data.image_dataset import ImageDataset
        dataset = ImageDataset(opt)
    elif name == 'image_generator_train_option' or name == 'image_generator_test_option':
        from data.image_edge_dataset import ImageEdgeDataset
        dataset = ImageEdgeDataset(opt)
    else:
        raise ValueError("不存在option [%s] 所对应的数据集" % name)

    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=opt.shuffle, num_workers=0)
    return dataloader
