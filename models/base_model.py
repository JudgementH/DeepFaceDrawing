import os
from abc import ABC, abstractmethod

import torch


class BaseModel(ABC):
    """模型抽象类
    需要重写方法
    - __init__
    - set_input
    - forward
    - optimize_parameters
    - name :决定保存位置
    """

    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device(
            'cpu')  # get device name: CPU or GPU
        self.save_dir = opt.checkpoints_dir
        self.loss_names = []
        self.model_names = []
        self.schedulers = []
        self.optimizers = []
        # self.visual_names = []
        # self.image_paths = []

    @abstractmethod
    def name(self):
        return "base_model"

    @abstractmethod
    def set_input(self, input):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def optimize_parameters(self):
        pass

    def update_learning_rate(self):
        """更新网络的学习率(每个epoch调用一次)"""
        for scheduler in self.schedulers:
            scheduler.step()

    def eval(self):
        """把本model内所有神经网络进入评估模式"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()

    def test(self):
        """在no_grad下运行一次forward"""
        with torch.no_grad():
            self.forward()

    def save_networks(self, epoch):
        """保存模型
        @:param epoch (int) -- 当前的 epoch; 保存到save_dir中的文件 '%s_%s.pth' % (model_name, epoch)
        """
        # for name in self.model_names:
        #     if isinstance(name, str):
        #         save_filename = '%s_%s.pth' % (epoch, name)
        #         save_path = os.path.join(self.save_dir, save_filename)
        #         net = getattr(self, name)
        #         if isinstance(net, torch.nn.DataParallel):
        #             torch.save(net.module.cpu().state_dict(), save_path)
        #         else:
        #             torch.save(net.cpu().state_dict(), save_path)
        #         if len(self.gpu_ids) > 0 and torch.cuda.is_available():
        #             net.cuda(self.gpu_ids[0])
        model_dict = {}
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                if isinstance(net, torch.nn.DataParallel):
                    model_dict[name] = net.module.cpu().state_dict()
                else:
                    model_dict[name] = net.cpu().state_dict()
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    net.cuda(self.gpu_ids[0])
        save_filename = '%s_%s.pth' % (self.name(), epoch)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(model_dict, save_path)

    def load_networks(self, epoch):
        """加载模型
        @:param epoch (int) -- load的目标epoch数; 读取save_dir中的文件 '%s_%s.pth' % (name, epoch)
        """
        # for name in self.model_names:
        #     if isinstance(name, str):
        #         load_filename = '%s_net_%s.pth' % (epoch, name)
        #         load_path = os.path.join(self.save_dir, load_filename)
        #         net = getattr(self, name)
        #         if isinstance(net, torch.nn.DataParallel):
        #             net = net.module
        #         print('正在读取模型，模型地址: %s' % load_path)
        #         state_dict = torch.load(load_path, map_location=self.device)
        #         if hasattr(state_dict, '_metadata'):
        #             del state_dict._metadata
        #         net.load_state_dict(state_dict)
        load_filename = '%s_%s.pth' % (self.name(), epoch)
        load_path = os.path.join(self.save_dir, load_filename)
        print("loading model with [%s]" % load_path)
        state_dict = torch.load(load_path, map_location=self.device)
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                net.load_state_dict(state_dict[name])

    def print_networks(self, verbose):
        """打印模型参数总数，if verbose then 打印模型架构
        @:param verbose (bool): 为True打印模型架构，False不打印
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] 参数总数 : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')
