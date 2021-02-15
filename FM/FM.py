import torch.nn as nn
from itertools import count
from collections import defaultdict
from scipy.sparse import csr
import numpy as np
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

"""
FM(Factorization Machine)主要是为了解决数据稀疏的情况下，特征怎样组合的问题。
已一个广告分类的问题为例，根据用户与广告位的一些特征，来预测用户是否会点击广告。
参考： https://www.pianshen.com/article/4561285296/
"""


# FM model
class FM_model(nn.Module):
    def __init__(self, p, k):
        super(FM_model, self).__init__()
        self.p = p    # 特征的数量
        self.k = k    # emb_dim
        self.linear = nn.Linear(self.p, 1, bias=True)
        self.v = nn.Parameter(torch.randn(self.k, self.p))  # 交叉特征向量

    def fm_layer(self, x):
        linear_part = self.linear(x)
        inter_part1 = torch.mm(x, self.v.t())
        inter_part2 = torch.mm(torch.pow(x, 2), torch.pow(self.v, 2).t())
        output = linear_part + 0.5 * torch.sum(torch.pow(inter_part1, 2) - inter_part2)
        return output

    def forward(self, x):
        output = self.fm_layer(x)
        return output


# 要使用FM模型，我们首先要将数据处理成一个矩阵，矩阵的大小是用户数* 电影数(样本数*特征总数)；通过使用scipy.sparse中的csr.csr_matrix实现
def vectorize_dic(dic, ix=None, p=None, n=0, g=0):  # n:num samples ; g: num groups
    """
    dic -- dictionary of feature lists. Keys are the name of features,keys:user,item
    ix -- index generator (default None)
    p -- dimension of feature space (number of columns in the sparse matrix) (default None)
    """
    if ix == None:
        ix = dict()  # value是列编号

    nz = n * g  # number of non-zeros

    col_ix = np.empty(nz, dtype=int)  # 存放的是行偏移量

    i = 0
    for k, lis in dic.items():  # user_id,item_id
        for t in range(len(lis)):
            ix[str(lis[t]) + str(k)] = ix.get(str(lis[t]) + str(k), 0) + 1  # 统计用户和商品出现的次数
            col_ix[i + t * g] = ix[str(lis[t]) + str(
                k)]  # 交叉存放元素,并且col_ix[i+t*g]表示该位置存放的是当前user_id 或 item_id 之前位置出现的总次数，因此，每个id会多次出现，并于原位置一一对应
        i += 1  # 偶数位：user_id , 技术位 ：item_id

    row_ix = np.repeat(np.arange(0, n), g)  # [0 0 1 1 2 2 3 3 4 4]   # 存放的是data中元素对应的列编号（可重复）
    data = np.ones(nz)  # 存放的是非0数据元素
    if p == None:
        p = len(ix)
    print(p)
    ixx = np.where(col_ix < p)  # np.where只有条件的，则输出满足条件元素的坐标,  结果是返回全部样本数据对应对应的index
    # 其中data，row_ind和col_ind满足关系a [row_ind [k]，col_ind [k]] = data [k]

    return csr.csr_matrix((data[ixx], (row_ix[ixx], col_ix[ixx])), shape=(n, p)), ix  # 按行对矩阵进行压缩，非零数值
    # 函数接收三个参数，第一个参数是数值，第二个参数是每个数对应的列号，第三个参数是每行的起始的偏移量
    # https://blog.csdn.net/u012871493/article/details/51593451


# 生成器
def batcher(X_, y_=None, batch_size=-1):
    n_samples = X_.shape[0]

    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
        raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))

    for i in range(0, n_samples, batch_size):
        upper_bound = min(i + batch_size, n_samples)
        ret_x = X_[i:upper_bound]
        ret_y = None
        if y_ is not None:
            ret_y = y_[i:i + batch_size]
            yield (ret_x, ret_y)


if __name__ == "__main__":
    # data proprecess
    cols = ['user', 'item', 'rating', 'timestamp']

    train = pd.read_csv('data/ua.base', delimiter='\t', names=cols)
    test = pd.read_csv('data/ua.test', delimiter='\t', names=cols)

    x_train, ix = vectorize_dic({'users': train['user'].values,
                                 'items': train['item'].values}, n=len(train.index), g=2)

    x_test, ix = vectorize_dic({'users': test['user'].values,
                                'items': test['item'].values}, ix, x_train.shape[1], n=len(test.index), g=2)

    y_train = train['rating'].values
    y_test = test['rating'].values

    x_train = x_train.todense()
    x_test = x_test.todense()
    print(x_train.shape)   # 第一维度为样本的数量，第二维度为user和item的unique的数量
    print(x_test.shape)
    # 用户数*电影数大小为待学习的fm矩阵
    # train
    n, p = x_train.shape
    k = 10
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FM_model(p, k).to(device)
    loss_fn = nn.MSELoss()
    optimer = torch.optim.SGD(model.parameters(), lr=0.0001, weight_decay=0.001)
    epochs = 100
    for epoch in range(epochs):
        loss_epoch = 0.0
        loss_all = 0.0
        perm = np.random.permutation(x_train.shape[0])
        model.train()
        for x, y in tqdm(batcher(x_train[perm], y_train[perm], batch_size)):
            model.zero_grad()
            x = torch.as_tensor(np.array(x.tolist()), dtype=torch.float, device=device)
            y = torch.as_tensor(np.array(y.tolist()), dtype=torch.float, device=device)
            x = x.view(-1, p)
            y = y.view(-1, 1)
            preds = model(x)
            loss = loss_fn(preds, y)
            loss_all += loss.item()
            loss.backward()
            optimer.step()
        loss_epoch = loss_all / len(x)
        print(f"Epoch [{epoch}/{10}], "
              f"Loss: {loss_epoch:.8f} ")
