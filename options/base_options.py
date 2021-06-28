import argparse
import os
import parser

import torch

from util import util


class BaseOption:
    """"用来控制模型参数的option的超类"""

    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        """"初始化parser"""

        # 基础信息
        self.parser.add_argument('--name', type=str, default='base_option', help='opt的名称，会决定opt的保存位置')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. -1代表使用CPU')
        # self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='模型保存根地址')

        self.initialized = True

    def print_options(self):
        """打印option
        把option保存入文本文件 / [checkpoints_dir] / {opt.name}_opt.txt
        """
        opt = self.opt
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = opt.checkpoints_dir
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.name))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """得到option对象, 创建文件夹, 设置gpu"""
        if not self.initialized:
            raise RuntimeError("未初始化，请先调用方法 initialize()")
        self.opt = self.parser.parse_args()

        self.print_options()

        # set gpu ids
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])
        return self.opt
