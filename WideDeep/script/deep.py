import torch
import torch.nn as nn
from utils import linear


# Sparse Features----> Dense Embedding----> Hidden Layers  ----> Output Units
# 输入特征主要分为两类，一类是数值特征（可直接输入DNN），一类是类别特征（需要经过Embedding之后才能输入到DNN中）:即稀疏特征转化为稠密向量
class DeepModel(nn.Module):
    def __init__(self, deep_columns_idx, embedding_columns_dict, hidden_layers, dropouts, output_dim):
        """
        init parameters
        :param deep_columns_idx: dict include column name and it's index
            e.g. {'age': 0, 'career': 1,...}
        :param embedding_columns_dict: dict include categories columns name and number of unique val and embedding dimension
            e.g. {'age':(10, 32),...}
        # 类比于word2vec,词汇表的权重矩阵，矩阵的大小：维度1为词汇表的长度，维度2为稠密向量的大小，因为每个特征代表的不同的意义，所以不同的类别系数特征需要被映射到不同的向量空间中，
        # 所以需要对不同的特征构建   embedding
        :param hidden_layers: number of hidden layers
        :param deep_columns_idx: dict of columns name and columns index
        :param dropouts: list of float each hidden layers dropout len(dropouts) == hidden_layers - 1
        """
        super(DeepModel, self).__init__()
        self.embedding_columns_dict = embedding_columns_dict
        self.deep_columns_idx = deep_columns_idx
        for key, val in embedding_columns_dict.items():
            setattr(self, 'dense_col_' + key, nn.Embedding(val[0], val[1]))  # 多特征并联的 dense embedding层
            """
            hasattr(obj,name):判断一个对象里面是否有name属性或者name方法
            getattr(obj,name[,default]): 获取对象obj中的属性和方法，
            setattr（obj,name,values）: 给对象的属性赋值，若属性不存在，则先创建在赋值
            """
        embedding_layer = 0
        for col in self.deep_columns_idx.keys():  # 这里表示的第一层神经元的个数最好采用2的n次幂相近的
            if col in embedding_columns_dict:
                embedding_layer += embedding_columns_dict[col][1]
            else:
                embedding_layer += 1
        self.layers = nn.Sequential()

        hidden_layers = [embedding_layer] + hidden_layers  # 论文里units的个数为最近的2的次1幂
        dropouts = [0.0] + dropouts
        for i in range(1, len(hidden_layers)):
            self.layers.add_module(
                'hidden_layer_{}'.format(i - 1),
                linear(hidden_layers[i - 1], hidden_layers[i], dropouts[i - 1])
            )
        self.layers.add_module('last_linear', nn.Linear(hidden_layers[-1], output_dim))

    def forward(self, x):
        emb = []
        continuous_cols = [col for col in self.deep_columns_idx.keys() if col not in self.embedding_columns_dict]
        for col, _ in self.embedding_columns_dict.items():
            if col not in self.deep_columns_idx:
                raise ValueError("ERROR column name may be your deep_columns_idx dict is not math the"
                                 "embedding_columns_dict")
            else:
                idx = self.deep_columns_idx[col]
                # 第一维度表示样本个数batch_size，第二维度inx表示该特征的具体取值，根据其值拿到权重矩阵中对应的embedding
                emb.append(getattr(self, 'dense_col_' + col)(x[:, idx].long()))

        for col in continuous_cols:
            idx = self.deep_columns_idx[col]
            emb.append(x[:, idx].view(-1, 1).float())
        embedding_layers = torch.cat(emb, dim=1)
        out = self.layers(embedding_layers)
        return out
