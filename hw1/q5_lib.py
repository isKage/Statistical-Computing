import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn

import matplotlib.pyplot as plt
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['font.sans-serif'] = 'SimHei'
from sklearn.metrics import accuracy_score

import time
from IPython import display
import warnings
warnings.filterwarnings('ignore')


# 加载 Fashion-MNIST 数据集
def load_data_fashion_mnist(batch_size, resize=None, root='./data', num_workers=4):
    trans = [transforms.ToTensor()]                 # 将图像数据从 PIL 格式转换为 32 位浮点数的 tensor 格式，并除以 255 使得所有像素的数值均在 0～1 之间
    if resize:
        trans.insert(0, transforms.Resize(resize))  # 图像大小转换
    trans = transforms.Compose(trans)               # 将转化列表组合为复合变换
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, transform=trans, download=True)
    return (DataLoader(mnist_train, batch_size, shuffle=True, num_workers=num_workers),
            DataLoader(mnist_test, batch_size, shuffle=False, num_workers=num_workers))


# 计时器
class Timer:
    
    # 记录多次运行时间
    def __init__(self):
        self.times = []

    # 启动计时器
    def start(self):
        self.tik = time.time()

    # 停止计时器并将时间记录在列表中
    def stop(self):
        self.times.append(time.time() - self.tik)

    # 返回当前记录的所有时间的总和
    def sum(self):
        return sum(self.times)

    # 返回平均时间
    def avg(self):
        return sum(self.times) / len(self.times)


# 动画绘制器
class Animator:
    
    # 增量地绘制图像
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None,
                 xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1, figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes,]
        
        # 捕获参数
        self.config_axes = lambda: self.configure_axes(xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    # 设置参数
    def configure_axes(self, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        ax = self.axes[0]
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        if legend:
            ax.legend(legend)
        ax.grid(True)

    # 向图像中添加多个数据点
    def add(self, x, y):
        if not hasattr(y, '__len__'):
            y = [y]
        n = len(y)
        if not hasattr(x, '__len__'):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


# 评估模型的准确率
def evaluate(model, data_iter, device):

    y_true = []  # 真实标签
    y_pred = []  # 预测标签
    
    model.eval()
    with torch.no_grad():
        for X, y in data_iter:
            
            # 模型预测
            X, y = X.to(device), y.to(device)
            y_hat = model(X)

            # 保存每个 batch 的真实标签、预测标签
            y_true.extend(y.cpu().tolist())
            y_pred.extend(y_hat.argmax(axis=1).cpu().tolist())

    # 计算准确率并返回
    return accuracy_score(y_true, y_pred)


# 模型训练与评估
def train(model, train_iter, test_iter, num_epochs, loss_fn, optimizer, device):

    # 指标初始化
    best_test_acc = 0
    
    # 可视化准备
    num_batches = len(train_iter)
    animator = Animator(xlabel='Epoch', xlim=[0, num_epochs], legend=['train loss', 'train acc', 'test acc'])
    timer = Timer()

    # 开始训练
    for epoch in range(num_epochs):
        model.train()
        metric = torch.tensor([0, 0, 0], dtype=float)  # 当前损失总和、当前正确个数、当前样本数
        for i, (X, y) in enumerate(train_iter):
            
            # 前向传播并计算损失
            timer.start()
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            timer.stop()
            
            # 更新 metric 并可视化
            acc_num = (y == y_hat.argmax(axis=1)).sum()
            metric += torch.tensor([loss * len(y), acc_num, len(y)])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (train_loss, train_acc, None))
        
        # 测试集评估与可视化
        test_acc = evaluate(model, test_iter, device)
        animator.add(epoch + 1, (None, None, test_acc))

        # 保存最佳模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            # torch.save(model.state_dict(), 'best_model.pth')

    # 输出最终训练结果
    print(f'train loss: {train_loss:.3f}, train acc: {train_acc:.3f}, best test acc: {best_test_acc:.3f}')
    print(f'{len(train_iter.dataset) * num_epochs / timer.sum():.1f} examples/sec')
