import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import argparse

torch.manual_seed(2021)
np.random.seed(2021)

labels = ['HandStart', 'FirstDigitTouch', 'BothStartLoadPhase', 'LiftOff',
          'Replace', 'BothReleased']

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

# 1. 首先定义模型结构（必须与训练时的结构完全相同）
class NNet(nn.Module):
    def __init__(self, in_channels=32, out_channels=6):
        super(NNet, self).__init__()
        self.hidden = 32
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 5, padding=2),
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

class EEGSignalDataset(Dataset):
    def __init__(self, data, gt, m=0, s=0, soft_label=True, train=True):
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


class EEGPredictor:
    def __init__(self, model_path='model.pt', device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.mean = None
        self.std = None

    def calculate_mean_std(self, data_list):
        """计算数据的均值和标准差"""
        if not isinstance(data_list, list):
            data_list = [data_list]

        # 将所有数据连接在一起
        all_data = np.concatenate([d.reshape(d.shape[0], -1) for d in data_list], axis=1)

        # 计算均值和标准差
        self.mean = np.mean(all_data, axis=1).reshape(-1, 1)
        self.std = np.std(all_data, axis=1).reshape(-1, 1)

        print("均值形状:", self.mean.shape)
        print("标准差形状:", self.std.shape)

        return self.mean, self.std

    def load_model(self, model_path):
        try:
            model = NNet()
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()  # 设置为评估模式
            model = model.to(self.device)
            print(f"模型已加载到 {self.device}")
            return model
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            return None

    def preprocess_data(self, data):
        """预处理输入数据"""
        if self.mean is None or self.std is None:
            # 如果均值和标准差未计算，则计算它们
            self.calculate_mean_std(data)

        # 标准化数据
        data = (data - self.mean) / (self.std + 1e-7)
        return data

    def predict(self, data_file):
        """预测单个文件的数据"""
        try:
            # 读取数据
            x = pd.read_csv(data_file)
            x = x.iloc[:, 2:-3].values  # 去掉第一列
            x = x.T.astype(np.float32)  # 转置并转换类型

            # 预处理数据
            x = self.preprocess_data(x)

            # 创建数据集和数据加载器
            dataset = EEGSignalDataset([x], [np.zeros((6, x.shape[1]))],
                                       train=False, soft_label=False)
            dataloader = DataLoader(dataset, batch_size=1024,
                                    num_workers=0, shuffle=False)

            predictions = []
            # 进行预测
            with torch.no_grad():
                for batch, _ in tqdm(dataloader):
                    batch = batch.to(self.device)
                    pred = self.model(batch).detach().cpu().numpy()
                    predictions.append(pred)

            # 合并预测结果
            predictions = np.concatenate(predictions, axis=0).squeeze(axis=-1)
            return predictions

        except Exception as e:
            print(f"预测过程出错: {str(e)}")
            return None


# 使用示例
if __name__ == "__main__":
    predictor = EEGPredictor('model.pt')

    # 预测单个文件
    test_file = 'openvibe-arm-lift.csv'
    predictions = predictor.predict(test_file)

    if predictions is not None:
        # 处理预测结果
        print("预测形状:", predictions.shape)

        # 可以将结果保存为CSV文件
        results_df = pd.DataFrame(predictions, columns=labels)
        results_df.to_csv('predictions.csv', index=False)