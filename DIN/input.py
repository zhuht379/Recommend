import numpy as np

"""
DataInput 从 train_set中读取数据，其中：
u:保存用户的user_id ，即代码中的reviewID，那么u就是用户id序列
i:表示正样本/负样本，后面正负样本统一用目标节点（target）来描述，即i为目标节点序列，
y:表示目标节点 的label，取值为1或0；
sl：保存用户历史行为序列的真实长度（代码中的len(t[1])就是求历史行为序列的长度），
max_sl : 表示序列中的最大长度

由于用户历史序列的长度是不固定的，因此引入hist_i，其为一个矩阵，将序列长度固定为max_sl，对于长度不足maxl_sl
的序列，使用0来进行填充（注意hist_i使用zero矩阵进行初始化的）
"""


class DataInput:
    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size  # epoch_num
        if self.epoch_size * self.batch_size < len(self.data):  # 向上取整
            self.epoch_size += 1
        self.i = 0  # 初始化迭代器的初始指针位置

    def __iter__(self):  # 返回实例本身，这里实例对象就是一个迭代器
        return self

    def __next__(self):
        if self.i == self.epoch_size:  # 最后一轮epoch，重新下一轮iter
            raise StopIteration
        ts = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size,
                                                     len(self.data))]  # 取 mini_batch_data
        self.i += 1  # 迭代器指针加1
        u, i, y, sl = [], [], [], []  # train_set=[user_id,hist,pos(neg)_item,label]
        for t in ts:  # mini_batch 中数据分离存放
            u.append(t[0])
            i.append(t[2])
            y.append(t[3])
            sl.append(len(t[1]))  # 用户真实商品行为的长度
        max_sl = max(sl)  # 每轮batch中最长的用户行为序列长度
        hist_i = np.zeros([len(ts), max_sl], np.int64)  # 用户—-浏览行为矩阵
        k = 0  # 矩阵行指针，代表着user_id 的index
        for t in ts:
            for l in range(len(t[1])):
                hist_i[k][l] = t[1][l]  # 将用户真实行为item_id(index) 存放到用户——浏览行为矩阵中
            k += 1
        return self.i, (u, i, y, hist_i, sl)  # i：label_item_list      y:0/1 list


class DataInputTest:
    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i == self.epoch_size:
            raise StopIteration
        ts = self.data[self.i * self.batch_size:min((self.i + 1) * self.batch_size, len(self.data))]
        self.i += 1
        u, i, j, sl = [], [], [], []
        for t in ts:
            # t 表示（reviewerID，hist，pos_list[i]/neg_list[i],1/0）
            # 其中 t[1]=hist ，为用户历史行为序列
            u.append(t[0])
            i.append(t[2][0])  # i： 为样本对应的正样本列表
            j.append(t[2][1])  # j: 样本对应的负样本列表
            sl.append(len(t[1]))
        max_sl = max(sl)
        hist_i = np.zeros([len(ts), max_sl], np.int64)
        k = 0
        for t in ts:
            for l in range(len(t[1])):
                hist_i[k][l] = t[1][l]
            k += 1
        return self.i, (u, i, j, hist_i, sl)

# 用户行为特征（user bahavior features）:一般是指某个用户对于多个商品的行为序列
