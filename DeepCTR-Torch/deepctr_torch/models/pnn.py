# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

from .basemodel import BaseModel
from ..inputs import combined_dnn_input
from ..layers import DNN, concat_fun, InnerProductLayer, OutterProductLayer

"""
在我们目前的重排序模型中，大概分为以下几类特征：
    1. deal(即团购单，下同)维度的特征：主要是deal本身的一些属性，包括价格、折扣、销量、评分、类别、点击率等
    2. user维度的特征：包括用户等级、用户的人口属性、用户的客户端类型等
    3. user、deal的交叉特征：包括用户对deal的点击、收藏、购买等
    4. 距离特征：包括用户的实时地理位置、常去地理位置、工作地、居住地等与poi的距离
    5. 对于非线性模型，上述特征可以直接使用；而对于线性模型，则需要对特征值做一些分桶、归一化等处理，使特征值成为0~1之间的连续值或01二值。
"""


class PNN(BaseModel):
    """
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param l2_reg_embedding: float . L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param use_inner: bool,whether use inner-product or not.
    :param use_outter: bool,whether use outter-product or not.
    :param kernel_type: str,kernel_type used in outter-product,can be ``'mat'`` , ``'vec'`` or ``'num'``
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :return: A PyTorch model instance.

    """

    def __init__(self, dnn_feature_columns, dnn_hidden_units=(128, 128), l2_reg_embedding=1e-5, l2_reg_dnn=0,
                 init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu', use_inner=True, use_outter=False,
                 kernel_type='mat', task='binary', device='cpu', ):

        super(PNN, self).__init__([], dnn_feature_columns, l2_reg_linear=0, l2_reg_embedding=l2_reg_embedding,
                                  init_std=init_std, seed=seed, task=task, device=device)

        if kernel_type not in ['mat', 'vec', 'num']:
            raise ValueError("kernel_type must be mat,vec or num")

        self.use_inner = use_inner  # 内积
        self.use_outter = use_outter  # 外积，非线性组合部分 https://blog.csdn.net/qq_18293213/article/details/90262378
        self.kernel_type = kernel_type
        self.task = task

        product_out_dim = 0
        num_inputs = self.compute_input_dim(dnn_feature_columns, include_dense=False, feature_group=True)
        num_pairs = int(num_inputs * (num_inputs - 1) / 2)

        if self.use_inner:
            product_out_dim += num_pairs
            self.innerproduct = InnerProductLayer(device=device)

        if self.use_outter:
            product_out_dim += num_pairs
            self.outterproduct = OutterProductLayer(
                num_inputs, self.embedding_size, kernel_type=kernel_type, device=device)

        self.dnn = DNN(product_out_dim + self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                       activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=False,
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
        linear_signal = torch.flatten(
            concat_fun(sparse_embedding_list), start_dim=1)

        if self.use_inner:
            inner_product = torch.flatten(
                self.innerproduct(sparse_embedding_list), start_dim=1)

        if self.use_outter:
            outer_product = self.outterproduct(sparse_embedding_list)

        if self.use_outter and self.use_inner:
            product_layer = torch.cat(
                [linear_signal, inner_product, outer_product], dim=1)
        elif self.use_outter:
            product_layer = torch.cat([linear_signal, outer_product], dim=1)
        elif self.use_inner:
            product_layer = torch.cat([linear_signal, inner_product], dim=1)
        else:
            product_layer = linear_signal

        dnn_input = combined_dnn_input([product_layer], dense_value_list)
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)
        logit = dnn_logit

        y_pred = self.out(logit)

        return y_pred
