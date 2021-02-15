# -*- coding:utf-8 -*-

import os, sys
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from time import time

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utils import data_preprocess
import torch.backends.cudnn

sys.path.append('../')


# 算法解读：
# 1. https://blog.csdn.net/qq_18293213/article/details/90262378
# 2. https://zhuanlan.zhihu.com/p/148985611


class PNN(torch.nn.Module):

    def __init__(self, field_size, feature_sizes, embedding_size=4, h_depth=3, deep_layers=[32, 32, 32],
                 is_deep_dropout=True, dropout_deep=[0.5, 0.5, 0.5], use_inner_product=True, use_outer_product=False,
                 deep_layers_activation='relu', n_epochs=64, batch_size=256, learning_rate=0.003, optimizer_type='adam',
                 is_batch_norm=False, verbose=False,
                 random_seed=1024, weight_decay=0.0, loss_type='logloss', eval_metric=roc_auc_score, use_cuda=True,
                 n_class=1, greater_is_better=True
                 ):
        super(PNN, self).__init__()
        self.field_size = field_size
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.h_depth = h_depth
        self.deep_layers = deep_layers
        self.is_deep_dropout = is_deep_dropout
        self.dropout_deep = dropout_deep
        self.use_inner_product = use_inner_product
        self.use_outer_product = use_outer_product
        self.deep_layers_activation = deep_layers_activation
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.is_batch_norm = is_batch_norm
        self.verbose = verbose
        self.weight_decay = weight_decay  # L2 正则
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.use_cuda = use_cuda
        self.n_class = n_class
        self.greater_is_better = greater_is_better
        torch.manual_seed(self.random_seed)

        # 是否使用GPu
        if self.use_cuda and not torch.cuda.is_available():
            self.use_cuda = False
            print("Cuda is not available, automatically changed into cpu model")

        # -------------- 第一层 Embedding layer --------------------------
        self.embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes])

        # ------------ first order part (linear part) --------------------
        # product layer层中的线性部分，也就是把Embedding层的特征concatenate组成一个向量
        self.first_order_weight = nn.ModuleList([nn.ParameterList(
            [torch.nn.Parameter(torch.randn(self.embedding_size), requires_grad=True) for j in range(self.field_size)])
            for i in range(self.deep_layers[0])])
        self.bias = torch.nn.Parameter(torch.randn(self.deep_layers[0]), requires_grad=True)

        # ------------ second order part (quadratic part) ---------------
        # 非线性组合部分 https://blog.csdn.net/qq_18293213/article/details/90262378
        if self.use_inner_product:  # 内积
            self.inner_second_weight_emb = nn.ModuleList([nn.ParameterList(
                [torch.nn.Parameter(torch.randn(self.embedding_size), requires_grad=True) for j in
                 range(self.field_size)]) for i in range(self.deep_layers[0])])

        if self.use_outer_product:  # 外积
            arr = []
            for i in range(self.deep_layers[0]):
                tmp = torch.randn(self.embedding_size, self.embedding_size)
                arr.append(torch.nn.Parameter(torch.mm(tmp, tmp.t())))
            self.outer_second_weight_emb = nn.ParameterList(arr)

        # ------------------- DNN ---------------------------------
        for i, h in enumerate(self.deep_layers[1:], 1):
            setattr(self, 'linear_' + str(i), nn.Linear(self.deep_layers[i - 1], self.deep_layers[i]))
            if self.is_batch_norm:
                setattr(self, 'batch_norm_' + str(i), nn.BatchNorm1d(deep_layers[i]))
            if self.is_deep_dropout:
                setattr(self, 'linear_' + str(i) + '_dropout', nn.Dropout(self.dropout_deep[i]))
        self.deep_last_layer = nn.Linear(self.deep_layers[-1], self.n_class)

    def forward(self, Xi, Xv):

        emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in enumerate(self.embeddings)]

        # first order part (linear part)
        first_order_arr = []
        for i, weight_arr in enumerate(self.first_order_weight):
            tmp_arr = []
            for j, weight in enumerate(weight_arr):
                tmp_arr.append(torch.sum(emb_arr[j] * weight, 1))
            first_order_arr.append(sum(tmp_arr).view([-1, 1]))
        first_order = torch.cat(first_order_arr, 1)

        # second order part (quadratic part)
        if self.use_inner_product:
            inner_product_arr = []
            for i, weight_arr in enumerate(self.inner_second_weight_emb):
                tmp_arr = []
                for j, weight in enumerate(weight_arr):
                    tmp_arr.append(torch.sum(emb_arr[j] * weight, 1))
                sum_ = sum(tmp_arr)
                inner_product_arr.append((sum_ * sum_).view([-1, 1]))
            inner_product = torch.cat(inner_product_arr, 1)
            first_order = first_order + inner_product

        if self.use_outer_product:
            outer_product_arr = []
            emb_arr_sum = sum(emb_arr)
            emb_matrix_arr = torch.bmm(emb_arr_sum.view([-1, self.embedding_size, 1]),
                                       emb_arr_sum.view([-1, 1, self.embedding_size]))
            for i, weight in enumerate(self.outer_second_weight_emb):
                outer_product_arr.append(torch.sum(torch.sum(emb_matrix_arr * weight, 2), 1).view([-1, 1]))
            outer_product = torch.cat(outer_product_arr, 1)
            first_order = first_order + outer_product

        # hidden layers
        if self.deep_layers_activation == 'sigmoid':
            activation = F.sigmoid
        elif self.deep_layers_activation == 'tanh':
            activation = F.tanh
        else:
            activation = F.relu
        x_deep = first_order
        for i, h in enumerate(self.deep_layers[1:], 1):
            x_deep = getattr(self, 'linear_' + str(i))(x_deep)
            if self.is_batch_norm:
                x_deep = getattr(self, 'batch_norm_' + str(i))(x_deep)
            x_deep = activation(x_deep)
            if self.is_deep_dropout:
                x_deep = getattr(self, 'linear_' + str(i) + '_dropout')(x_deep)
        x_deep = self.deep_last_layer(x_deep)
        return torch.sum(x_deep, 1)

    def fit(self, Xi_train, Xv_train, y_train, Xi_valid=None, Xv_valid=None,
            y_valid=None, ealry_stopping=False, refit=False, save_path=None):
        # Xi_train: feat_index是特征的一个序号，主要用于通过embedding_lookup选择我们的embedding
        # Xv_train: 是对应的特征值，如果是离散特征的话，就是1，如果不是离散特征的话，就保留原来的特征值
        """
        :param Xi_train: [[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]
                        indi_j is the feature index of feature field j of sample i in the training set
        :param Xv_train: [[val1_1, val1_2, ...], [val2_1, val2_2, ...], ..., [vali_1, vali_2, ..., vali_j, ...], ...]
                        vali_j is the feature value of feature field j of sample i in the training set
                        vali_j can be either binary (1/0, for binary/categorical features) or float (e.g., 10.24, for numerical features)
        :param y_train: label of each sample in the training set

        """

        if save_path and not os.path.exists('/'.join(save_path.split('/')[0:-1])):
            print("Save path is not existed!")
            return
        is_valid = False
        # ----------------- 训练集 --------------------
        Xi_train = np.array(Xi_train).reshape((-1, self.field_size, 1))
        Xv_train = np.array(Xv_train)
        y_train = np.array(y_train)
        x_size = Xi_train.shape[0]

        # ----------------- 验证集 ----------------------
        if Xi_valid:
            Xi_valid = np.array(Xi_valid).reshape((-1, self.field_size, 1))
            Xv_valid = np.array(Xv_valid)
            y_valid = np.array(y_valid)
            x_valid_size = Xi_valid.shape[0]
            is_valid = True

        # train model
        model = self.train()  # 定义模型
        # 优化器选择
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'rmsp':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'adag':
            optimizer = torch.optim.Adagrad(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        # 定义损失函数
        criterion = F.binary_cross_entropy_with_logits

        train_result = []
        valid_result = []
        for epoch in range(self.n_epochs):
            total_loss = 0.0
            batch_iter = x_size // self.batch_size
            epoch_begin_time = time()
            batch_begin_time = time()
            for i in range(batch_iter + 1):
                offset = i * self.batch_size
                end = min(x_size, offset + self.batch_size)
                if offset == end:
                    break
                batch_xi = Variable(torch.LongTensor(Xi_train[offset:end]))
                batch_xv = Variable(torch.FloatTensor(Xv_train[offset:end]))
                batch_y = Variable(torch.FloatTensor(y_train[offset:end]))
                if self.use_cuda:
                    batch_xi, batch_xv, batch_y = batch_xi.cuda(), batch_xv.cuda(), batch_y.cuda()
                optimizer.zero_grad()
                outputs = model(batch_xi, batch_xv)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.data[0]
                if self.verbose:
                    if i % 100 == 99:  # print every 100 mini-batches
                        eval = self.evaluate(batch_xi, batch_xv, batch_y)
                        print('[%d, %5d] loss: %.6f metric: %.6f time: %.1f s' %
                              (epoch + 1, i + 1, total_loss, eval, time() - batch_begin_time))
                        total_loss = 0.0
                        batch_begin_time = time()
            # 模型每训练完一个epoch，对模型评估一次效果
            train_loss, train_eval = self.eval_by_batch(Xi_train, Xv_train, y_train, x_size)
            train_result.append(train_eval)
            print('*' * 50)
            print('[%d] loss: %.6f metric: %.6f time: %.1f s' %
                  (epoch + 1, train_loss, train_eval, time() - epoch_begin_time))
            print('*' * 50)
            # 在验证集上验证模型效果
            if is_valid:
                valid_loss, valid_eval = self.eval_by_batch(Xi_valid, Xv_valid, y_valid, x_valid_size)
                valid_result.append(valid_eval)
                print('*' * 50)
                print('[%d] loss: %.6f metric: %.6f time: %.1f s' %
                      (epoch + 1, valid_loss, valid_eval, time() - epoch_begin_time))
                print('*' * 50)
            # 保存将每一轮epoch训练好的模型
            if save_path:
                torch.save(self.state_dict(), save_path)
            # 训练提前终止
            if is_valid and ealry_stopping and self.training_termination(valid_result):
                print("early stop at [%d] epoch!" % (epoch + 1))
                break

        # fit a few more epoch on train+valid until result reaches the best_train_score
        if is_valid and refit:
            if self.verbose:
                print("refitting the model")
            if self.greater_is_better:
                best_epoch = np.argmax(valid_result)
            else:
                best_epoch = np.argmin(valid_result)
            best_train_score = train_result[best_epoch]
            Xi_train = np.concatenate((Xi_train, Xi_valid))
            Xv_train = np.concatenate((Xv_train, Xv_valid))
            y_train = np.concatenate((y_train, y_valid))
            x_size = x_size + x_valid_size
            self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
            for epoch in range(64):
                batch_iter = x_size // self.batch_size
                for i in range(batch_iter + 1):
                    offset = i * self.batch_size
                    end = min(x_size, offset + self.batch_size)
                    if offset == end:
                        break
                    batch_xi = Variable(torch.LongTensor(Xi_train[offset:end]))
                    batch_xv = Variable(torch.FloatTensor(Xv_train[offset:end]))
                    batch_y = Variable(torch.FloatTensor(y_train[offset:end]))
                    if self.use_cuda:
                        batch_xi, batch_xv, batch_y = batch_xi.cuda(), batch_xv.cuda(), batch_y.cuda()
                    optimizer.zero_grad()
                    outputs = model(batch_xi, batch_xv)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                train_loss, train_eval = self.eval_by_batch(Xi_train, Xv_train, y_train, x_size)
                if save_path:
                    torch.save(self.state_dict(), save_path)
                if abs(best_train_score - train_eval) < 0.001 or \
                        (self.greater_is_better and train_eval > best_train_score) or \
                        ((not self.greater_is_better) and train_result < best_train_score):
                    break
            if self.verbose:
                print("refit finished")

    def eval_by_batch(self, Xi, Xv, y, x_size):
        total_loss = 0.0
        y_pred = []
        batch_size = 16384
        batch_iter = x_size // batch_size
        criterion = F.binary_cross_entropy_with_logits
        model = self.eval()
        for i in range(batch_iter + 1):
            offset = i * batch_size
            end = min(x_size, offset + batch_size)
            if offset == end:
                break
            batch_xi = Variable(torch.LongTensor(Xi[offset:end]))
            batch_xv = Variable(torch.FloatTensor(Xv[offset:end]))
            batch_y = Variable(torch.FloatTensor(y[offset:end]))
            if self.use_cuda:
                batch_xi, batch_xv, batch_y = batch_xi.cuda(), batch_xv.cuda(), batch_y.cuda()
            outputs = model(batch_xi, batch_xv)
            pred = F.sigmoid(outputs).cpu()
            y_pred.extend(pred.data.numpy())
            loss = criterion(outputs, batch_y)
            total_loss += loss.data[0] * (end - offset)
        total_metric = self.eval_metric(y, y_pred)
        return total_loss / x_size, total_metric

    # shuffle three lists simutaneously
    def shuffle_in_unison_scary(self, a, b, c):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)

    def training_termination(self, valid_result):
        if len(valid_result) > 4:
            if self.greater_is_better:
                if (valid_result[-1] < valid_result[-2]) and \
                        (valid_result[-2] < valid_result[-3]) and \
                        (valid_result[-3] < valid_result[-4]):
                    return True
            else:
                if (valid_result[-1] > valid_result[-2]) and \
                        (valid_result[-2] > valid_result[-3]) and \
                        (valid_result[-3] > valid_result[-4]):
                    return True
        return False

    def predict(self, Xi, Xv):
        """
        :param Xi: the same as fit function
        :param Xv: the same as fit function
        :return: output, ont-dim array
        """
        Xi = np.array(Xi).reshape((-1, self.field_size, 1))
        Xi = Variable(torch.LongTensor(Xi))
        Xv = Variable(torch.FloatTensor(Xv))
        if self.use_cuda and torch.cuda.is_available():
            Xi, Xv = Xi.cuda(), Xv.cuda()

        model = self.eval()
        pred = F.sigmoid(model(Xi, Xv)).cpu()
        return (pred.data.numpy() > 0.5)

    def predict_proba(self, Xi, Xv):
        Xi = np.array(Xi).reshape((-1, self.field_size, 1))
        Xi = Variable(torch.LongTensor(Xi))
        Xv = Variable(torch.FloatTensor(Xv))
        if self.use_cuda and torch.cuda.is_available():
            Xi, Xv = Xi.cuda(), Xv.cuda()

        model = self.eval()
        pred = F.sigmoid(model(Xi, Xv)).cpu()
        return pred.data.numpy()

    def inner_predict(self, Xi, Xv):

        model = self.eval()
        pred = F.sigmoid(model(Xi, Xv)).cpu()
        return (pred.data.numpy() > 0.5)

    def inner_predict_proba(self, Xi, Xv):

        model = self.eval()
        pred = F.sigmoid(model(Xi, Xv)).cpu()
        return pred.data.numpy()

    def evaluate(self, Xi, Xv, y):

        y_pred = self.inner_predict_proba(Xi, Xv)
        return self.eval_metric(y.cpu().data.numpy(), y_pred)


if __name__ == '__main__':
    result_dict = data_preprocess.read_criteo_data('../data/train.csv', '../data/category_emb.csv')
    test_dict = data_preprocess.read_criteo_data('../data/test.csv', '../data/category_emb.csv')
    with torch.cuda.device(2):
        pnn = PNN(39, result_dict['feature_sizes'], batch_size=128 * 64, verbose=True, use_cuda=True,
                  weight_decay=0.00001,
                  use_inner_product=True, use_outer_product=True).cuda()
        pnn.fit(result_dict['index'], result_dict['value'], result_dict['label'],
                test_dict['index'], test_dict['value'], test_dict['label'], ealry_stopping=True, refit=False,
                save_path='../data/model/pnn.pkl')

"""
在我们目前的重排序模型中，大概分为以下几类特征：
    1. deal(即团购单，下同)维度的特征：主要是deal本身的一些属性，包括价格、折扣、销量、评分、类别、点击率等
    2. user维度的特征：包括用户等级、用户的人口属性、用户的客户端类型等
    3. user、deal的交叉特征：包括用户对deal的点击、收藏、购买等
    4. 距离特征：包括用户的实时地理位置、常去地理位置、工作地、居住地等与poi的距离
    5. 对于非线性模型，上述特征可以直接使用；而对于线性模型，则需要对特征值做一些分桶、归一化等处理，使特征值成为0~1之间的连续值或01二值。
"""
