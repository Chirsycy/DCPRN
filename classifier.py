# 分类器卷积内容！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

import torch.nn as nn


class network(nn.Module):
    def __init__(self, numclass, feature_extractor):  # 初始化函数 参数：分类的数量 特征提取器
        super(network, self).__init__()  # 调用父类nn.Module的初始化函数。
        self.a = 0
        self.feature = feature_extractor
        self.fc = nn.Linear(256, numclass, bias=True)

    def forward(self, input):
        x = self.feature(input)  # 使用特征提取器提取输入的特征。
        x = self.fc(x)  # 将提取到的特征输入到全连接层中进行分类。
        return x  # 返回分类结果

    def Incremental_learning(self, numclass):
        self.a = 0
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features

        self.fc = nn.Linear(in_feature, numclass, bias=True)

        self.fc.weight.data[:out_feature] = weight[:out_feature]
        self.fc.bias.data[:out_feature] = bias[:out_feature]

    def feature_extractor(self, inputs):
        return self.feature(inputs)
