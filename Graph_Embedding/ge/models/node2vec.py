# -*- coding:utf-8 -*-
from gensim.models import Word2Vec
import pandas as pd

from ..walker import RandomWalker

"""
https://blog.csdn.net/u012151283/article/details/87081272
https://zhuanlan.zhihu.com/p/46344860
1.同一个社区内的节点表示相似
2.拥有类似结构特征的节点表示相似
"""
class Node2Vec:

    """
    node2vec引入两个超参数p和q来控制随机游走的策略，假设当前随机游走经过边（t,v）到达顶点v
    Return parameter p:参数p控制重复访问刚刚访问过顶点的概率。
    注意到p仅作用与dts=0的情况，而dts=0表示顶点x就是当前顶点v之前刚刚访问过的顶点，若p较较高，则访问刚刚访问过的顶点的概率会变低，反之变高

    in-out parameter q：
    q控制着游走是向外还是向内，若q>1，随机游走倾向于访问和t接近的顶点（偏向BFS）。若q<1。倾向于访问远离t的顶点（偏向DFS）
    """
    def __init__(self, graph, walk_length, num_walks, p=1.0, q=1.0, workers=1, use_rejection_sampling=0):

        self.graph = graph          # 有向（有权）图
        self._embeddings = {}
        self.walker = RandomWalker(
            graph, p=p, q=q, use_rejection_sampling=use_rejection_sampling)   # 实例化RandomWalk 类

        print("Preprocess transition probs...")
        # 转移概率
        self.walker.preprocess_transition_probs()

        self.sentences = self.walker.simulate_walks(
            num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1)

    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):

        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = embed_size
        kwargs["sg"] = 1
        kwargs["hs"] = 0  # node2vec not use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["iter"] = iter

        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done!")

        self.w2v_model = model

        return model

    def get_embeddings(self, ):
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        for word in self.graph.nodes():
            self._embeddings[word] = self.w2v_model.wv[word]

        return self._embeddings
