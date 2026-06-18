#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

# ResNet models: support both FlexFL original and SmartFL version
from models.resnet import ResNet18_cifar  # FlexFL original ResNet (default)
from models.resnet_smart import ResNet18_cifar as ResNet18_SmartFL  # SmartFL ResNet
from models.mobileNetV2 import MobileNetV2
from models.vgg import vgg_16_bn
from models.vit_flexfl import vit_small_flexfl, vit_small_scalefl
from models.Nets import CNNCifar, CNNMnist, ModelFlexFL, CNNFashionMnist
from models.Update import *
from models.test import test_img
from models.Fed import Aggregation
from models.LSTM import CharLSTM
from models.test import test
