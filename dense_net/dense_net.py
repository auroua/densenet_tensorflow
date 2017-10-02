import tensorflow as tf
import numpy as np
import logging
from dense_net_utils import stack_blocks


class DenseNet(object):
    def __init__(self, inputs):
        self.inputs = inputs

    def densenet121(self):
        net_parameter = [(32, 6), (32, 12), (32, 24), (32, 16)]
        return stack_blocks(self.inputs, net_parameter)

    def densenet169(self):
        net_parameter = [(32, 6), (32, 12), (32, 32), (32, 32)]
        return stack_blocks(self.inputs, net_parameter)

    def densenet201(self):
        net_parameter = [(32, 6), (32, 12), (32, 48), (32, 32)]
        return stack_blocks(self.inputs, net_parameter)

    def densenet161(self):
        net_parameter = [(48, 6), (48, 12), (48, 36), (48, 24)]
        return stack_blocks(self.inputs, net_parameter)

    def densenetL40k12(self):
        net_parameter = [(12, 12), (12, 12), (12, 12)]
        return stack_blocks(self.inputs, net_parameter)

    def densenetL100k12(self):
        net_parameter = [(32, 12), (32, 12), (32, 12)]
        return stack_blocks(self.inputs, net_parameter)

    def densenetL100k24(self):
        net_parameter = [(32, 24), (32, 24), (32, 24)]
        return stack_blocks(self.inputs, net_parameter)

    def densenetL100k12_bc(self):
        net_parameter = [(32, 12), (32, 12), (32, 12)]
        return stack_blocks(self.inputs, net_parameter)

    def densenetL250k24_bc(self):
        net_parameter = [(82, 24), (82, 24), (82, 24)]
        return stack_blocks(self.inputs, net_parameter)

    def densenetL190k40_bc(self):
        net_parameter = [(62, 40), (62, 40), (62, 40)]
        return stack_blocks(self.inputs, net_parameter)


if __name__ == '__main__':
    pass