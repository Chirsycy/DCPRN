# 增量学习算法的模型训练
# 引入必要python库
import os
import random

import numpy as np
import scipy
import scipy.io as io
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

# 引入自己编写的神经网络模型
from classifier import network


class protoAugSSL:
    def __init__(self, args, all_classes, map_reverse, class_map, file_name, feature_extractor, task_size, device):
        self.file_name = file_name
        self.args = args
        self.all_classes = all_classes
        self.map_reverse = map_reverse
        self.class_map = class_map
        self.epochs = args.epochs1
        self.learning_rate = args.learning_rate
        self.model = network(args.fg_nc, feature_extractor)
        self.radius = 0
        self.prototype = None
        self.class_label = None
        self.std = None
        self.numclass = args.fg_nc
        self.task_size = task_size
        self.device = device
        self.old_model = None
        self.old_old_model = None
        self.margin = 20
        self.sc_T = 1
        self.kd_T = 2

        self.alpha = 1
        self.lloss = 0

        self.train_dataset = io.loadmat(
            'E:/DATA/BHS/BHS_slices/process_houston_ratio_9.mat')
        self.test_dataset = io.loadmat(
            'E:/DATA/BHS/BHS_slices/process_houston_ratio_9.mat')
        self.train_data = None
        self.train_label = None
        self.test_data = []
        self.test_label = []
        self.a = 0

    def beforeTrain(self, current_task):
        print("self.numclass", self.numclass)
        self.model.eval()
        classes = self.all_classes[:self.numclass] if current_task == 0 else self.all_classes[self.args.fg_nc + (
                current_task - 1) * self.task_size: self.args.fg_nc + current_task * self.task_size]
        self.train_data, self.train_label = self.getTrainData(classes)
        print("self.train_label:", self.train_label)
        self.getTestData(classes)

        if current_task > 0:
            self.a = 1
            print("增量self.numclass：", self.numclass)
            self.model.Incremental_learning(self.numclass)

        self.model.train()
        self.model.cuda()

    def train(self, current_task, class_map, old_class=0):  # class_map将类别映射到数字标签
        if old_class == 0:
            self.epochs = self.args.epochs1
            print("无旧类原型。")
        else:
            self.epochs = self.args.epochs2
            pro = np.array(self.prototype)
            print("旧类的原型：", pro.shape)  # 打印旧类原型的形状

        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=2e-5)
        for epoch in range(1, self.epochs):
            n = 0
            for i in range(self.train_data.shape[0]):
                if (i + 1) % self.args.batch_size == 0:
                    images, target = self.train_data[n:i + 1].transpose((0, 3, 1, 2)), self.train_label[n:i + 1]
                    n += self.args.batch_size
                    images = torch.as_tensor(torch.from_numpy(images), dtype=torch.float32)
                    seen_labels = torch.LongTensor([class_map[label] for label in target])
                    target = Variable(seen_labels)

                    if torch.cuda.is_available():
                        images, target = images.cuda(), target.cuda()
                    if old_class == 0:
                        images = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], 1)
                        images = images.view(-1, 20, 9, 9)
                        target = torch.stack([target for _ in range(4)], 1).view(-1)

                    opt.zero_grad()
                    loss = self._compute_loss(images, target, old_class)
                    loss.backward()
                    opt.step()

            if (self.train_data.shape[0] - n) < self.args.batch_size and (self.train_data.shape[0] - n) != 0:
                images, target = self.train_data[n:].transpose((0, 3, 1, 2)), self.train_label[n:]
                images = torch.as_tensor(Variable(torch.from_numpy(images)), dtype=torch.float32)
                seen_labels = torch.LongTensor([class_map[label] for label in target])
                target = Variable(seen_labels)
                if torch.cuda.is_available():
                    images, target = images.cuda(), target.cuda()
                if old_class == 0:
                    images = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], 1).view(-1, 20, 9, 9)
                    target = torch.stack([target for _ in range(4)], 1).view(-1)
                loss = self._compute_loss(images, target, old_class)
                loss.backward()
                opt.step()

            if epoch % self.args.print_freq == 0:  # 每隔self.args.print_freq输出一次
                accuracy = self._test()
                print('epoch:%d, accuracy:%.5f, loss:%.5f' % (epoch, accuracy, self.lloss.data))
        self.protoSave(self.model, current_task)

    def _test(self):  # 两个参数：测试数据的数据加载器 旧类别的数量
        self.model.eval()
        self.test_data, self.test_label = np.array(self.test_data), np.array(self.test_label)
        correct, total = 0.0, 0.0
        n = 0
        for i in range(self.test_data.shape[0]):
            if (i + 1) % self.args.batch_size == 0:
                imgs, labels = self.test_data[n:i + 1].transpose((0, 3, 1, 2)), self.test_label[n:i + 1]
                n += self.args.batch_size
                imgs = torch.as_tensor(torch.from_numpy(imgs), dtype=torch.float32).to(self.device)
                labels = torch.as_tensor(torch.from_numpy(labels), dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    outputs = self.model(imgs)

                predicts = torch.max(outputs, dim=1)[1]
                predicts = torch.tensor([self.map_reverse[pred] for pred in predicts.cpu().numpy()])

                correct += (predicts == labels.cpu()).sum()
                total += len(labels)

        if (self.test_data.shape[0] - n) < self.args.batch_size and (self.test_label.shape[0] - n) != 0:
            imgs, labels = self.test_data[n:].transpose((0, 3, 1, 2)), self.test_label[n:]
            imgs = torch.as_tensor(torch.from_numpy(imgs), dtype=torch.float32).to(self.device)
            labels = torch.as_tensor(torch.from_numpy(labels), dtype=torch.float32).to(self.device)

            with torch.no_grad():
                outputs = self.model(imgs)

            predicts = torch.max(outputs, dim=1)[1]
            predicts = torch.tensor([self.map_reverse[pred] for pred in predicts.cpu().numpy()])

            correct += (predicts == labels.cpu()).sum()
            total += len(labels)

        accuracy = correct.item() / total
        self.model.train()
        return accuracy

    def _compute_loss(self, imgs, target, old_class=0):  # 参数：输入图像 目标标签  旧类别的数量
        target = torch.as_tensor(target, dtype=torch.long)
        target = target.cuda()
        feature = self.model.feature(imgs)  # 仅是特征提取部分，不包括最后全连接层  # torch.Size([100, 512])
        feature = feature.cuda()

        if self.old_model is None:
            output = self.model(imgs)
            output = output.cuda()
            self.args.temp = torch.tensor([1]).cuda()  # 1
            loss_cls = nn.CrossEntropyLoss()(output / self.args.temp, target)

            self.lloss = loss_cls
            return loss_cls
        else:
            proto_aug = []
            proto_aug_label = []
            index = list(range(old_class))
            np.random.shuffle(index)

            for i in index:
                for _ in range(self.args.batch_size):
                    temp_avg = np.average(self.prototype[i])
                    temp_std = self.std[i]
                    temp = self.prototype[i] + np.random.normal(temp_avg, temp_std, 256)
                    proto_aug.append(temp)
                    proto_aug_label.append(self.class_label[i])

            proto_aug = torch.tensor(proto_aug, dtype=torch.float32, device=self.device)
            proto_aug_label = torch.tensor(proto_aug_label, dtype=torch.long, device=self.device)

            final_features = torch.cat((feature, proto_aug), dim=0)

            final_output = self.model.fc(final_features)

            target = target.view(-1, 1)
            proto_aug_label = proto_aug_label.view(-1, 1)

            final_labels = torch.cat((target, proto_aug_label), dim=0).squeeze(1)

            self.args.temp = torch.tensor([1], device=self.device)  # 1
            loss_cls = nn.CrossEntropyLoss()(final_output / self.args.temp, final_labels)

            output_images_old = self.old_model(imgs).cuda()
            output_images = final_output[:feature.shape[0], :output_images_old.shape[1]]
            feature_old = self.old_model.feature(imgs).cuda()

            self.kd_T = 2
            loss_kd1 = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(feature / self.kd_T, dim=0),
                                                           F.softmax(feature_old / self.kd_T, dim=0)) * (
                               self.kd_T * self.kd_T)

            loss_kd2 = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output_images / self.kd_T, dim=0),
                                                           F.softmax(output_images_old / self.kd_T, dim=0)) * (
                               self.kd_T * self.kd_T)

            loss_kd = self.args.kd_weight1 * loss_kd1 + self.args.kd_weight2 * loss_kd2
            self.lloss = loss_kd + self.args.cls_weight * loss_cls
            return loss_cls + loss_kd

    def afterTrain(self):
        path = self.args.save_path + self.file_name + '/'
        os.makedirs(path, exist_ok=True)
        self.numclass += self.task_size
        filename = f"{path}{self.numclass - self.task_size}_model.pkl"
        torch.save(self.model, filename)
        self.old_model = torch.load(filename)
        self.old_model.to(self.device)
        self.old_model.eval()

    def protoSave(self, model, current_task):  # 参数：模型 数据集 当前任务编号
        model.eval()
        with torch.no_grad():
            n = 0  # 追踪当前处理的数据索引
            t = 0  # 判断是否首次处理数据
            for i in range(self.train_data.shape[0]):
                if (i + 1) % self.args.batch_size == 0:
                    images, target = self.train_data[n:i + 1].transpose((0, 3, 1, 2)), self.train_label[n:i + 1]
                    n += self.args.batch_size
                    images = torch.as_tensor(Variable(torch.from_numpy(images)), dtype=torch.float32)
                    seen_labels = torch.LongTensor([self.class_map[label] for label in target])
                    target = Variable(seen_labels)

                    if torch.cuda.is_available():
                        images, target = images.cuda(), target.cuda()
                    feature = model.feature(images)  # 进行特征提取
                    if feature.shape[0] == self.args.batch_size:  # 保证批次处理的一致性
                        if t == 0:
                            labels = target.cpu().detach().numpy()
                            features = feature.cpu().detach().numpy()
                            t = 1
                        else:
                            labels = np.append(labels, target.cpu().detach().numpy())
                            features = np.concatenate((features, feature.cpu().detach().numpy()))
            if (self.train_data.shape[0] - n) < self.args.batch_size and (self.train_label.shape[0] - n) != 0:
                images, target = self.train_data[n:].transpose((0, 3, 1, 2)), self.train_label[n:]
                images = torch.as_tensor(Variable(torch.from_numpy(images)), dtype=torch.float32)
                seen_labels = torch.LongTensor([self.class_map[label] for label in target])
                target = Variable(seen_labels).cuda()
                if torch.cuda.is_available():
                    images, target = images.cuda(), target.cuda()
                feature = model.feature(images)
                if t == 0:
                    labels = target.cpu().detach().numpy()
                    features = feature.cpu().detach().numpy()
                    t = 1
                else:
                    labels = np.append(labels, target.cpu().detach().numpy())
                    features = np.concatenate((features, feature.cpu().detach().numpy()))
        labels_set = []  # 用于存储唯一的标签类别
        for i in range(labels.shape[0]):  # 遍历标签数组的每个元素
            if labels[i] not in labels_set:  # 如果当前标签不在labels_set列表中将当前标签添加到labels_set列表中。
                labels_set.append(labels[i])
        print('当前任务的类别:', labels_set)
        feature_dim = features.shape[1]  # features.shape(48, 256)

        prototype = []  # 存储原型向量
        radius = []  # 存储半径值
        class_label = []  # 存储类别标签
        std_list = []  # 存储标准差值
        for item in labels_set:
            index = np.where(item == labels)[0]
            class_label.append(item)
            feature_classwise = features[index]
            std_list.append(np.std(feature_classwise))
            prototype.append(np.mean(feature_classwise, axis=0))
            if current_task == 0:
                cov = np.cov(feature_classwise.T)
                radius.append(np.trace(cov) / feature_dim)

        if current_task == 0:
            oc = np.mean(prototype, axis=0)
            self.prototype = [proto + scipy.spatial.distance.cosine(proto, oc) for proto in prototype]
            self.class_label = class_label
            self.std = std_list
            self.radius = np.sqrt(np.mean(radius))
        else:
            oc = np.sum(prototype + self.prototype, axis=0) / (len(prototype) + len(self.prototype))
            self.prototype = np.array([proto + scipy.spatial.distance.cosine(proto, oc) for proto in prototype] +
                                      [proto + scipy.spatial.distance.cosine(proto, oc) for proto in self.prototype])
            self.class_label = np.concatenate((class_label, self.class_label), axis=0)
            self.std = np.concatenate((std_list, self.std), axis=0)

        print("截止到当前所有的类别 self.class_label:", self.class_label)

    def random_sample(self, data, label):
        train_index = list(range(data.shape[0]))
        random.shuffle(train_index)
        random_data, random_label = [], []  # 存储打乱顺序后的数据和label
        for i in train_index:
            random_data.append(data[i])
            random_label.append(label[i])
        random_data = np.array(random_data)
        random_label = np.array(random_label)
        return random_data, random_label

    def getTrainData(self, classes):
        train_input, train_output = self.train_dataset['train_data'], self.train_dataset['train_labels']
        datas, labels = [], []
        for label in classes:
            for j in range(train_output.shape[1]):
                if train_output[0][j] == label:
                    datas.append(train_input[j])
                    labels.append(label)
        TrainData, TrainLabels = np.array(datas), np.array(labels)
        random_train_data, random_train_label = self.random_sample(TrainData, TrainLabels)
        return random_train_data, random_train_label

    def getTestData(self, classes):
        test_input, test_output = self.test_dataset['train_data'], self.test_dataset['train_labels']
        datas, labels = [], []
        print("classes", classes)
        for label in classes:
            for j in range(test_output.shape[1]):
                if test_output[0][j] == label:
                    data = test_input[j]
                    datas.append(data)
                    labels.append(label)
        datas, labels = np.array(datas), np.array(labels)
        self.test_data = datas if self.test_data == [] else np.concatenate((self.test_data, datas), axis=0)
        self.test_label = labels if self.test_label == [] else np.concatenate((self.test_label, labels), axis=0)
        self.test_data, self.test_label = self.random_sample(self.test_data, self.test_label)
