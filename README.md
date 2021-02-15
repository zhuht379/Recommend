# recommend

AI算法工程师手册：
1. 框架中文文档教程： http://docs.apachecn.org/
2. 电子数据及相关文档：https://www.bookstack.cn/explore
3. 各种算法的tensorFlow实现：https://codechina.csdn.net/mirrors/princewen/tensorflow_practice
4. https://zhuanlan.zhihu.com/p/270918998

FNN,PNN缺点：对于低阶的组合特征，学习到的比较少，而低阶特征对于CTR也非常的重要。
wide&deep：仍然需要人工特征工程
DeepFM优点：
       1.不需要预训练FM得到隐向量
       2. 不需要人工特征工程
       3.能同时学习低阶和高阶的组合特征
       4. FM模块和Deep模块共享Feature embedding 部分，可以更快的训练，以及更精准的训练学习。
与Wide&deep不同的是，DeepFM是端到端的训练，不需要人工特征工程

AUC反映整体样本间的排序能力，而点击率预估需要的是对每个用户来说所有产品的点击率排序，GAUC（group AUC）就是基于此提出的，能够解决线下AUC
猛涨，但线上效果变差的问题。https://blog.csdn.net/hnu2012/article/details/87892368

排序模型对来自不同通道的所有召回内容进行排序。在feeds中，内容是按照排序模型的输出自上而下显示，在广告推荐中，选择ctr最高
的若干个进行展示。除了合并通道的原因以外，召回和排序两个阶段所使用的特征也大不相同。

