import numpy as np
from zipfile import ZipFile
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import random
import torch.nn.functional as F
import argparse
from sklearn import metrics
from tqdm import tqdm
import gc
import shutil

FILE_PATH = 'input/grasp-and-lift-eeg-detection'
list_dir = os.listdir(FILE_PATH)

for zipfile in list_dir:
    with ZipFile(os.path.join(FILE_PATH, zipfile), 'r') as z:
        z.extractall()

labels = ['HandStart', 'FirstDigitTouch', 'BothStartLoadPhase', 'LiftOff',
       'Replace', 'BothReleased']

# 设置PyTorch和NumPy的随机种子，确保实验可重复性
torch.manual_seed(2021)
np.random.seed(2021)

# 检查是否有可用的GPU，如果有则使用GPU，否则使用CPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 创建命令行参数解析器
parser = argparse.ArgumentParser()

# 添加各种命令行参数
# 训练轮数设置，默认为1轮
parser.add_argument("--n_epochs", type=int, default=1, help="number of epochs of training")

# 批次大小设置，默认为1024
parser.add_argument("--batch_size", type=int, default=1024, help="size of the batches")

# Adam优化器的学习率，默认为0.002
parser.add_argument("--lr", type= float, default=0.002, help="adam's learning rate")

# Adam优化器的beta1参数，用于一阶矩估计，默认为0.5
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")

# Adam优化器的beta2参数，用于二阶矩估计，默认为0.99
parser.add_argument("--b2", type=float, default=0.99, help="adam: decay of first order momentum of gradient")

# 用于批次生成的CPU线程数，默认为1
parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")

# 神经网络输入长度，默认为2^10=1024
parser.add_argument("--in_len", type=int, default=2**10, help="length of the input fed to neural net")

# 输入信号通道数，默认为32
parser.add_argument("--in_channels", type=int, default=32, help="number of signal channels")

# 输出类别数，默认为6
parser.add_argument("--out_channels", type=int, default=6, help="number of classes")

# 数据分块大小，默认为1000
parser.add_argument("--chunk", type=int, default=1000, help="length of splited chunks")

# 解析命令行参数
# opt包含所有已定义的参数
# unknown包含所有未定义的参数
opt, unknown = parser.parse_known_args()

# 打印使用的设备类型（CPU或GPU）
print(device)



def read_csv(data, events):
    """
    读取数据和事件CSV文件的函数

    参数:
    data: 数据文件路径
    events: 事件文件路径

    返回:
    x: 数据矩阵
    y: 事件标签矩阵
    """
    # 读取数据和事件CSV文件
    x = pd.read_csv(data)
    y = pd.read_csv(events)

    # 从文件名中提取ID，去掉最后一个下划线后的部分 "hello_world_123"返回 "hello_world"
    id = '_'.join(x.iloc[0, 0].split('_')[:-1])

    # 提取数据矩阵，去掉第一列（通常是索引或时间戳）
    x = x.iloc[:, 1:].values

    # 提取事件矩阵，去掉第一列
    y = y.iloc[:, 1:].values

    return x, y


# 初始化训练数据集列表和对应的真实标签列表
trainset = []
gt = []  # ground truth（真实标签）

# 遍历训练数据目录中的所有文件
for filename in tqdm(os.listdir('./train')):  # tqdm用于显示进度条
    # 只处理包含'data'的文件
    if 'data' in filename:
        # 构建数据文件的完整路径
        data_file_name = os.path.join('./train', filename)

        # 从文件名中提取ID
        id = filename.split('.')[0]

        # 构建对应的事件文件名
        # 将数据文件名中的'_data'替换为'_events'
        events_file_name = os.path.join('./train', '_'.join(id.split('_')[:-1]) + '_events.csv')

        # 读取数据和事件文件
        x, y = read_csv(data_file_name, events_file_name)

        # 将数据添加到训练集列表中
        # .T进行转置，astype将数据类型转换为float32
        trainset.append(x.T.astype(np.float32))

        # 将标签添加到真实标签列表中
        gt.append(y.T.astype(np.float32))

# 从训练集中分离出验证集，取最后两个样本作为验证集
valid_dataset = trainset[-2:]  # 验证数据集
valid_gt = gt[-2:]  # 验证集标签
trainset = trainset[:-2]  # 移除验证集后的训练数据集
gt = gt[:-2]  # 移除验证集后的训练集标签

# 加载预计算的均值和标准差
# 这些通常用于数据标准化
# m = np.load('../input/cnn-eeg/mean.npy')  # 加载均值
# s = np.load('../input/cnn-eeg/std.npy')  # 加载标准差

def calculate_mean_std(data_list):
    # 将所有数据连接在一起
    all_data = np.concatenate([d.reshape(d.shape[0], -1) for d in data_list], axis=1)
    # 计算均值和标准差
    m = np.mean(all_data, axis=1)
    s = np.std(all_data, axis=1)
    return m, s

# 使用训练集计算均值和标准差
m, s = calculate_mean_std(trainset)
# 将均值和标准差转换为列向量形状
m = m.reshape(-1, 1)
s = s.reshape(-1, 1)

def resample_data(gt, chunk_size=opt.chunk):
    """
    将长信号切分成更小的数据块，丢弃没有事件的数据块

    参数:
    gt: ground truth数据（标签）
    chunk_size: 数据块大小，默认使用配置中的chunk值

    返回:
    index: 保留的数据块索引列表
    """
    total_discard_chunks = 0  # 记录丢弃的数据块总数
    mean_val = []  # 存储每个数据块的平均值
    threshold = 0.01  # 事件密度阈值
    index = []  # 存储要保留的数据点的索引

    # 遍历每个样本
    for i in range(len(gt)):
        # 按chunk_size大小划分数据
        for j in range(0, gt[i].shape[1], chunk_size):
            # 计算当前数据块的平均值（事件密度）
            mean_val.append(np.mean(gt[i][:, j:min(gt[i].shape[1], j + chunk_size)]))

            # 如果事件密度低于阈值，丢弃该数据块
            if mean_val[-1] < threshold:
                total_discard_chunks += 1
            else:
                # 保留该数据块的所有时间点索引
                index.extend([(i, k) for k in range(j, min(gt[i].shape[1], j + chunk_size))])

    # 可视化数据块的事件密度分布
    plt.plot([0, len(mean_val)], [threshold, threshold], color='r')  # 绘制阈值线
    plt.scatter(range(len(mean_val)), mean_val, s=1)  # 绘制每个数据块的事件密度
    plt.show()

    # 打印统计信息
    print('Total number of chunks discarded: {} chunks'.format(total_discard_chunks))
    print('{}% data'.format(total_discard_chunks / len(mean_val)))

    # 清理内存
    del mean_val
    gc.collect()
    return index


class EEGSignalDataset(Dataset):
    """
    脑电图信号数据集类，继承自PyTorch的Dataset类
    用于加载和预处理脑电图数据
    """

    def __init__(self, data, gt, m=m, s=s, soft_label=True, train=True):
        """
        初始化数据集

        参数:
        data: 脑电图原始数据
        gt: ground truth标签
        m: 均值，用于标准化
        s: 标准差，用于标准化
        soft_label: 是否使用软标签
        train: 是否为训练模式
        """
        self.data = data
        self.gt = gt
        self.train = train
        self.soft_label = soft_label
        self.eps = 1e-7  # 避免除零错误的小量

        # 根据是否为训练模式选择不同的数据索引方式
        if train:
            # 训练模式下使用resample_data函数筛选有效数据块
            self.index = resample_data(gt)
        else:
            # 测试模式下使用所有数据点
            self.index = [(i, j) for i in range(len(data)) for j in range(data[i].shape[1])]

        # 对数据进行标准化处理
        for dt in self.data:
            dt -= m  # 减去均值
            dt /= s + self.eps  # 除以标准差

    def __getitem__(self, i):
        """
        获取单个数据样本

        参数:
        i: 索引值

        返回:
        raw_data: 处理后的脑电图数据
        label: 对应的标签
        """
        # 获取数据索引
        i, j = self.index[i]

        # 截取指定长度的数据段和对应的标签
        raw_data, label = self.data[i][:, max(0, j - opt.in_len + 1):j + 1], self.gt[i][:, j]

        # 如果数据长度不足，进行零填充
        pad = opt.in_len - raw_data.shape[1]
        if pad:
            raw_data = np.pad(raw_data, ((0, 0), (pad, 0)), 'constant', constant_values=0)

        # 将数据转换为PyTorch张量
        raw_data, label = torch.from_numpy(raw_data.astype(np.float32)), torch.from_numpy(label.astype(np.float32))

        # 如果使用软标签，将小于0.02的标签值设为0.02
        if self.soft_label:
            label[label < .02] = .02

        return raw_data, label

    def __len__(self):
        """
        返回数据集的总长度
        """
        return len(self.index)


# 创建训练数据集实例
dataset = EEGSignalDataset(trainset, gt)

# 创建数据加载器
# batch_size: 每批处理的样本数
# num_workers: 用于数据加载的进程数
# shuffle: 是否随机打乱数据
dataloader = DataLoader(dataset, batch_size=opt.batch_size,num_workers=0, shuffle=True)

# 打印数据集大小
print(len(dataset))


class NNet(nn.Module):
    def __init__(self, in_channels=opt.in_channels, out_channels=opt.out_channels):
        super(NNet, self).__init__()
        self.hidden = 32
        self.net = nn.Sequential(
            nn.Conv1d(opt.in_channels, opt.in_channels, 5, padding=2),
            nn.Conv1d(self.hidden, self.hidden, 16, stride=16),
            nn.LeakyReLU(0.1),
            nn.Conv1d(self.hidden, self.hidden, 7, padding=3),
        )
        for i in range(6):
            self.net.add_module('conv{}'.format(i), self.__block(self.hidden, self.hidden))
        self.net.add_module('final', nn.Sequential(
            nn.Conv1d(self.hidden, out_channels, 1),
            nn.Sigmoid()
        ))

    def __block(self, inchannels, outchannels):
        return nn.Sequential(
            nn.MaxPool1d(2, 2),
            nn.Dropout(p=0.1, inplace=True),
            nn.Conv1d(inchannels, outchannels, 5, padding=2),
            nn.LeakyReLU(0.1),
            nn.Conv1d(outchannels, outchannels, 5, padding=2),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        return self.net(x)


# 初始化模型及训练设置
nnet = NNet()  # 创建神经网络实例
nnet.to(device)  # 将模型移动到指定设备（GPU/CPU）
loss_fnc = nn.BCELoss()  # 二元交叉熵损失函数
# 初始化Adam优化器
adam = optim.Adam(nnet.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
loss_his, train_loss = [], []  # 用于记录损失值的列表

# 设置模型为训练模式
nnet.train()

# 开始训练循环
for epoch in range(opt.n_epochs):
    p_bar = tqdm(dataloader)  # 创建进度条
    for i, (x, y) in enumerate(p_bar):
        # 将数据移动到指定设备
        x, y = x.to(device), y.to(device)

        # 前向传播
        pred = nnet(x)
        # 计算损失
        loss = loss_fnc(pred.squeeze(dim=-1), y)

        # 反向传播
        adam.zero_grad()  # 清空梯度
        loss.backward()  # 计算梯度
        adam.step()  # 更新参数

        # 记录损失值
        train_loss.append(loss.item())
        # 更新进度条描述
        p_bar.set_description('[Loss: {}]'.format(train_loss[-1]))

        # 每50批次记录一次平均损失
        if i % 50 == 0:
            loss_his.append(np.mean(train_loss))
            train_loss.clear()

    # 打印每个epoch的训练信息
    print('[Epoch {}/{}] [Loss: {}]'.format(epoch + 1, opt.n_epochs, loss_his[-1]))

# 保存模型
torch.save(nnet.state_dict(), 'model.pt')

# 绘制损失曲线
plt.plot(range(len(loss_his)), loss_his, label='loss')
plt.legend()
plt.show()

# 测试阶段
# 创建测试数据集和数据加载器
testset = EEGSignalDataset(valid_dataset, valid_gt, train=False, soft_label=False)
testloader = DataLoader(testset, batch_size=opt.batch_size,
                        num_workers=0, shuffle=False)

# 设置模型为评估模式
nnet.eval()
y_pred = []

# 在不计算梯度的情况下进行预测
with torch.no_grad():
    for x, _ in tqdm(testloader):
        x = x.to(device)
        pred = nnet(x).detach().cpu().numpy()
        y_pred.append(pred)


def plot_roc(y_true, y_pred):
    """
    绘制ROC曲线

    参数:
    y_true: 真实标签
    y_pred: 预测值
    """
    # 创建3x2的子图
    fig, axs = plt.subplots(3, 2, figsize=(15, 13))
    for i, label in enumerate(labels):
        # 计算每个类别的ROC曲线
        fpr, tpr, _ = metrics.roc_curve(y_true[i], y_pred[i])
        ax = axs[i // 2, i % 2]
        # 绘制ROC曲线
        ax.plot(fpr, tpr)
        ax.set_title(label + " ROC")
        # 绘制对角线
        ax.plot([0, 1], [0, 1], 'k--')

    plt.show()


# 处理验证集预测结果
# 连接所有预测结果，并调整维度
y_pred = np.concatenate(y_pred, axis=0).squeeze(axis=-1)
# 连接所有真实标签
valid_gt = np.concatenate(valid_gt, axis=1)
# 绘制ROC曲线
plot_roc(valid_gt, y_pred.T)
# 计算并打印验证集的ROC AUC分数
print('auc roc: ', metrics.roc_auc_score(valid_gt.T, y_pred))

# 清理内存
del y_pred
del testset
del testloader
del valid_dataset
del valid_gt
gc.collect()  # 强制垃圾回收

# # 对训练集进行预测和评估
# y_pred = []
# y_true = []
# # 在不计算梯度的情况下进行预测
# with torch.no_grad():
#     for x, y in tqdm(dataloader):
#         x = x.to(device)
#         # 获取预测结果并转换为numpy数组
#         pred = nnet(x).squeeze(dim=-1).detach().cpu().numpy()
#         y_pred.append(pred)
#         y_true.append(y)
#
# # 连接所有预测结果
# y_pred = np.concatenate(y_pred, axis=0)
# # 连接所有真实标签
# y_true = np.concatenate(y_true, axis=0)
# # 将小于0.1的标签值设为0（二值化处理）
# y_true[y_true < .1] = 0
# # 绘制训练集的ROC曲线
# plot_roc(y_true.T, y_pred.T)
# # 计算并打印训练集的ROC AUC分数
# print('auc roc: ', metrics.roc_auc_score(y_true, y_pred))
#
# del y_pred
# del testset
# del testloader
# del valid_dataset
# del valid_gt
# gc.collect()
#
# y_pred = []
# y_true = []
# with torch.no_grad():
#     for x, y in tqdm(dataloader):
#         x = x.to(device)
#         pred = nnet(x).squeeze(dim=-1).detach().cpu().numpy()
#         y_pred.append(pred)
#         y_true.append(y)
#
# y_pred = np.concatenate(y_pred, axis=0)
# y_true = np.concatenate(y_true, axis=0)
# y_true[y_true<.1]=0
# plot_roc(y_true.T, y_pred.T)
# print('auc roc: ', metrics.roc_auc_score(y_true, y_pred))
#
# i = 1231
# with torch.no_grad():
#     input = dataset[i][0].unsqueeze(dim=0)
#     print(nnet(input.to(device)))
#     print(dataset[i][1])
#
# del y_pred
# del y_true
# del dataset
# del dataloader
# del trainset
# del gt
# gc.collect()
#
#
# class EEGSignalTestset(Dataset):
#     def __init__(self, data, m, s):
#         self.data = data
#         self.eps = 1e-7
#         self.data -= m
#         self.data /= s + self.eps
#
#     def __getitem__(self, i):
#         raw_data = self.data[:, max(0, i - opt.in_len + 1):i + 1]
#
#         pad = opt.in_len - raw_data.shape[1]
#         if pad:
#             raw_data = np.pad(raw_data, ((0, 0), (pad, 0)), 'constant', constant_values=0)
#
#         raw_data = torch.from_numpy(raw_data.astype(np.float32))
#         return raw_data
#
#     def __len__(self):
#         return self.data.shape[1]
#
#
# testset = []
# trial_len = {}
# FNAME = "./test/subj{}_series{}_{}.csv"
#
# for subj in range(1, 13):
#     for series in [9, 10]:
#         data_file_name = FNAME.format(subj, series, 'data')
#         x = pd.read_csv(data_file_name).iloc[:, 1:].values
#         testset.append(x.T.astype(np.float32))
#         trial_len['{}_{}'.format(subj, series)] = testset[-1].shape[-1]
#
# testset = np.concatenate(testset, axis=1)
#
# testset = EEGSignalTestset(testset, m, s)
# dataloader = DataLoader(testset, batch_size = opt.batch_size,num_workers = opt.n_cpu, shuffle=False)
#
# y_pred = []
# with torch.no_grad():
#     for x in tqdm(dataloader):
#         x = x.to(device)
#         pred = nnet(x).detach().cpu().numpy()
#         y_pred.append(pred)
#
# y_pred = np.concatenate(y_pred, axis=0).squeeze(axis=-1)
#
# submission = pd.DataFrame(y_pred, index=\
#     ['subj{}_series{}_{}'.format(sbj, i, j) for sbj in range(1,13) for i in [9,10] for j in range(trial_len['{}_{}'.format(sbj, i)])],\
#                          columns=labels)
# submission.to_csv('Submission.csv',index_label='id',float_format='%.3f')
#
# a = pd.read_csv('Submission.csv')
# a.tail()
#
# try:
#     shutil.rmtree('./train')
#     shutil.rmtree('./test')
#     os.remove('sample_submission.csv')
# except:
#     pass