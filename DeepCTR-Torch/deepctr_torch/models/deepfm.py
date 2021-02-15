# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com
Reference:
    [1] Guo H, Tang R, Ye Y, et al. Deepfm: a factorization-machine based neural network for ctr prediction[J]. arXiv preprint arXiv:1703.04247, 2017.(https://arxiv.org/abs/1703.04247)
"""
import torch
import torch.nn as nn

from .basemodel import BaseModel
from ..inputs import combined_dnn_input
from ..layers import FM, DNN
# DeepFM中，FM算法负责对一阶特征以及以及特征的两两组合而成的二阶特征进行特征的提取，DNN算法负责对输入的一阶特征进行全连接等操作 形成的高阶特征进行特征的提取。
# FM 和Deep两部分共享输入参数和Embedding，对于每一个特征i，存在一个数值型参数W_i 用于描述特征的一阶重要性，同时还存在一个隐向量V_i用于描述该特征和其他特征的交叉信息，V_i同时喂给FM和Deep
# 分别用于挖掘二阶交叉信息和高阶交叉信息，整个模型采用联合训练的方式对上述参数进行学习：y=sigmoid(y_FM+y_DNN )
class DeepFM(BaseModel):
    """Instantiates the DeepFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param use_fm: bool,use FM part or not
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :return: A PyTorch model instance.
    
    """
    # deep 这里embedding层这输入层之间权重矩阵就是有FM模型中的隐向量V构成，之前的做法是先对FM进行预训练得到一个训练好的隐向量V
    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, use_fm=True,
                 dnn_hidden_units=(256, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                 dnn_dropout=0,
                 dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu'):

        super(DeepFM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                     device=device)

        self.use_fm = use_fm
        self.use_dnn = len(dnn_feature_columns) > 0 and len(
            dnn_hidden_units) > 0
        if use_fm:
            self.fm = FM()

        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            self.dnn_linear = nn.Linear(
                dnn_hidden_units[-1], 1, bias=False).to(device)

            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)
        self.to(device)

    def forward(self, X):

        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        logit = self.linear_model(X)

        if self.use_fm and len(sparse_embedding_list) > 0:
            fm_input = torch.cat(sparse_embedding_list, dim=1)
            logit += self.fm(fm_input)

        if self.use_dnn:
            dnn_input = combined_dnn_input(
                sparse_embedding_list, dense_value_list)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)
            logit += dnn_logit

        y_pred = self.out(logit)

        return y_pred

# k:代表隐向量维数
# n可以代表离散值one-hot后的全部维数，也可以是n个field，每个域中取xi不为0的数
"""
对于离散特征的处理，我们使用的是将特征转换成为one-hot的形式，但是将One-hot类型的特征输入到DNN中，会导致网络参数太多：
one-hot作为DNN输入的问题：CTR预估任务里不可行，
解决思路：从oneHOt到Dense Vector，对不同的feature进行映射为embedding feature field,再加上两层全连接，让dense vector进行组合


"""