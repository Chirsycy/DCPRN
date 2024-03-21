import argparse
import warnings
import numpy as np
import torch
import torch.utils.data

from model import protoAugSSL
from feature_extraction import resnet18_cbam

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(
        description='Prototype Augmentation and Self-Supervision for Incremental Learning')
    parser.add_argument('--epochs1', default=101, type=int, help='Total number of epochs to run')
    parser.add_argument('--epochs2', default=201, type=int, help='Total number of epochs to run')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training')
    parser.add_argument('--print_freq', default=10, type=int, help='print frequency (default: 10)')
    parser.add_argument('--data_name', default='Houston', type=str, help='Dataset name to use')
    parser.add_argument('--total_nc', default=15, type=int, help='class number for the dataset')  # 数据集的类别
    parser.add_argument('--fg_nc', default=8, type=int, help='the number of classes in first task')
    parser.add_argument('--task_num', default=1, type=int, help='the number of incremental steps')  # 增量步长
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--protoAug_weight', default=10.0, type=float, help='protoAug loss weight')
    parser.add_argument('--kd_weight1', default=1, type=float, help='knowledge distillation loss weight')
    parser.add_argument('--kd_weight2', default=1, type=float, help='knowledge distillation loss weight')
    parser.add_argument('--cls_weight', default=1, type=float, help='knowledge distillation loss weight')
    parser.add_argument('--temp', default=1, type=float, help='training time temperature')
    parser.add_argument('--gpu', default='0', type=str, help='GPU id to use')

    parser.add_argument('--save_path', default='model_saved_check/', type=str, help='save files directory')

    args = parser.parse_args()
    print(args)

    cuda_index = 'cuda:' + args.gpu
    device = torch.device(cuda_index if torch.cuda.is_available() else "cpu")

    task_size = int((args.total_nc - args.fg_nc) / args.task_num)

    file_name = args.data_name + '_' + str(args.fg_nc) + '_' + str(args.task_num) + '_' + str(task_size)

    feature_extractor = resnet18_cbam()

    total_classes = args.total_nc
    print("total_class:", total_classes)
    perm_id = [3, 9, 10, 11, 5, 13, 6, 4, 0, 14, 7, 2, 12, 1, 8]
    print("perm_id:", perm_id)

    all_classes = np.array(perm_id)
    print("all_classes:", all_classes)
    class_map = {val: idx for idx, val in enumerate(perm_id)}
    map_reverse = {v: k for k, v in class_map.items()}

    model = protoAugSSL(args, all_classes, map_reverse, class_map, file_name, feature_extractor, task_size, device)
    for i in range(args.task_num + 1):
        if i == 0:
            print('#######################初次训练阶段###################')
            old_class = 0
        else:
            old_class = len(all_classes[:args.fg_nc + (i - 1) * task_size])
            print('#######################增量学习阶段###################')
        print('旧类的数量:', old_class)
        model.beforeTrain(i)
        model.train(i, class_map, old_class=old_class)
        model.afterTrain()


if __name__ == "__main__":
    main()
