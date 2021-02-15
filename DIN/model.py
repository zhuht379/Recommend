import tensorflow as tf


# 算法解读：https://zhuanlan.zhihu.com/p/139417423

class Model(object):
    # B:batch_size
    def __init__(self, user_count, item_count, cate_count, cate_list, predict_batch_size, predict_ads_num):
        self.u = tf.placeholder(tf.int32, [None, ])  # [B]           # self.u 用户id序列，大小为mini_batch的大小
        self.i = tf.placeholder(tf.int32, [None, ])  # [B]           # 正样本的item ，目标节点序列
        # 其中model中关于self.j的代码，它只会在运行测试集的时候会被用到，这和TF的静态图特性有关
        self.j = tf.placeholder(tf.int32, [None, ])  # [B]           # 负样本的item
        self.y = tf.placeholder(tf.float32, [None, ])  # [B]         # 目标节点对应的label序列，正样本对应1，负样本对应0
        self.hist_i = tf.placeholder(tf.int32, [None, None])  # [B, T]       # 用户行为特征（user Behavior）中的item序列，
        self.sl = tf.placeholder(tf.int32, [None, ])  # [B]                    # 记录用户行为序列的真实长度
        self.lr = tf.placeholder(tf.float64, [])

        hidden_units = 128

        """
        第0部分： 嵌入向量层：商品，类目将one_hot表示转化为稠密向量表示
        """
        user_emb_w = tf.get_variable("user_emb_w", [user_count,
                                                    hidden_units])  # user_id 的embedding weight (W 矩阵)
        item_emb_w = tf.get_variable("item_emb_w", [item_count,
                                                    hidden_units // 2])  # item_id的embedding weight （W 矩阵）
        item_b = tf.get_variable("item_b", [item_count],
                                 initializer=tf.constant_initializer(0.0))
        cate_emb_w = tf.get_variable("cate_emb_w", [cate_count, hidden_units // 2])  # cate_id 的embedding weight (W 矩阵)

        """ 映射向量查询"""
        cate_list = tf.convert_to_tensor(cate_list, dtype=tf.int64)  # cate_list 存放顺序是按照商品index的对应位置存放的
        """
        tf.gather(params,indices,axis=0):从params的axis维根据indices的参数值获取请切片
        因此这里的self.i 为商品的index，作用：拿到商品的cate信息
        """
        ic = tf.gather(cate_list, self.i)  # ic是item到category的转换 ,从cate_list中取出样本的cate。

        """
        第一部分：正样本的embedding，这里为正样本的embedding映射层，即从one-hot 转变为embedding的的层
        """
        # 正样本的embedding，正样本包括item 和 cate , 也即候选广告ad的embedding
        i_emb = tf.concat(values=[
            tf.nn.embedding_lookup(item_emb_w, self.i),
            tf.nn.embedding_lookup(cate_emb_w, ic),
        ], axis=1)  # 拼接长度为hidden_units形成最终的item_embedding
        i_b = tf.gather(item_b, self.i)  # 候选广告 wx+b 偏置项 （标量）

        """
        第二部分：负样本的embedding
        """
        # 负样本的embedding，负样本包括item 和cate。负样本候选ad的embedding
        jc = tf.gather(cate_list, self.j)  # 从cate_list中取出负样本的cate
        j_emb = tf.concat([
            tf.nn.embedding_lookup(item_emb_w, self.j),
            tf.nn.embedding_lookup(cate_emb_w, jc),
        ], axis=1)
        j_b = tf.gather(item_b, self.j)

        """
        第三部分：用户行为矩阵中行为商品的embedding
        """
        # 用户_行为商品的矩阵中每个商品的embedding
        hc = tf.gather(cate_list, self.hist_i)
        h_emb = tf.concat([
            tf.nn.embedding_lookup(item_emb_w, self.hist_i),
            tf.nn.embedding_lookup(cate_emb_w, hc),
        ], axis=2)  # 用户行为序列（user behavior）的embedding，包括item序列和cate序列

        """
        第四部分：得到用户浏览行为的sum_pooling embedding     w_i * embedding_i
        """
        # 候选商品ads和用户行为的对正样本的attention，activate操作，
        hist_i = attention(i_emb, h_emb, self.sl)  # (B,H)   (B,T,H)
        # -- attention end ---

        hist_i = tf.layers.batch_normalization(inputs=hist_i)
        hist_i = tf.reshape(hist_i, [-1, hidden_units], name='hist_bn')
        hist_i = tf.layers.dense(hist_i, hidden_units, name='hist_fcn')

        u_emb_i = hist_i  # attention之后为128维度的向量（也即是多个用户行为商品的sum_pooling ）

        """
        第五部分： 
        """
        hist_j = attention(j_emb, h_emb, self.sl)
        # -- attention end ---

        # hist_j = tf.layers.batch_normalization(inputs = hist_j)
        hist_j = tf.layers.batch_normalization(inputs=hist_j, reuse=True)
        hist_j = tf.reshape(hist_j, [-1, hidden_units], name='hist_bn')
        hist_j = tf.layers.dense(hist_j, hidden_units, name='hist_fcn', reuse=True)

        u_emb_j = hist_j

        # 打印attention后的embedding信息
        print(u_emb_i.get_shape().as_list())
        print(u_emb_j.get_shape().as_list())
        print(i_emb.get_shape().as_list())
        print(j_emb.get_shape().as_list())

        # -- fcn begin -------
        """
        召回部分的候选商品,  计算候选商品和用户行为商品的关系得分
        """
        din_i = tf.concat([u_emb_i, i_emb, u_emb_i * i_emb],
                          axis=-1)  # u_emb * i_emb 用户兴趣和候选广告的交叉特征，行为一维特征，商品一维特征，以及组合特征    [B,3H]
        din_i = tf.layers.batch_normalization(inputs=din_i, name='b1')
        d_layer_1_i = tf.layers.dense(din_i, 80, activation=tf.nn.sigmoid, name='f1')  # [B,80]
        d_layer_2_i = tf.layers.dense(d_layer_1_i, 40, activation=tf.nn.sigmoid, name='f2')  # [B,40]
        d_layer_3_i = tf.layers.dense(d_layer_2_i, 1, activation=None, name='f3')  # [B,] 每个样本得到一个数

        """
        负样本部分
        """
        din_j = tf.concat([u_emb_j, j_emb, u_emb_j * j_emb], axis=-1)
        din_j = tf.layers.batch_normalization(inputs=din_j, name='b1', reuse=True)
        d_layer_1_j = tf.layers.dense(din_j, 80, activation=tf.nn.sigmoid, name='f1', reuse=True)
        d_layer_2_j = tf.layers.dense(d_layer_1_j, 40, activation=tf.nn.sigmoid, name='f2', reuse=True)
        d_layer_3_j = tf.layers.dense(d_layer_2_j, 1, activation=None, name='f3', reuse=True)
        d_layer_3_i = tf.reshape(d_layer_3_i, [-1])
        d_layer_3_j = tf.reshape(d_layer_3_j, [-1])
        x = i_b - j_b + d_layer_3_i - d_layer_3_j  # [B]

        self.logits = i_b + d_layer_3_i  # w*x +b 得到logit    正样本softmax之后大于0.5，负样本的的分支小于0.5

        # prediciton for selected items
        # logits for selected item:
        item_emb_all = tf.concat([  # 商品矩阵和类目向量矩阵的拼接
            item_emb_w,
            tf.nn.embedding_lookup(cate_emb_w, cate_list)
        ], axis=1)  # 所有商品的embedding 列表
        item_emb_sub = item_emb_all[:predict_ads_num, :]  # 取前100个商品的embedding子列表 [100,H]
        item_emb_sub = tf.expand_dims(item_emb_sub, 0)  # [1,100,H]
        item_emb_sub = tf.tile(item_emb_sub, [predict_batch_size, 1, 1])  # [B,100,H],predict_batch_size =batch_size
        hist_sub = attention_multi_items(item_emb_sub, h_emb, self.sl)
        # -- attention end ---

        hist_sub = tf.layers.batch_normalization(inputs=hist_sub, name='hist_bn', reuse=tf.AUTO_REUSE)
        # print hist_sub.get_shape().as_list()
        hist_sub = tf.reshape(hist_sub, [-1, hidden_units])
        hist_sub = tf.layers.dense(hist_sub, hidden_units, name='hist_fcn', reuse=tf.AUTO_REUSE)

        u_emb_sub = hist_sub
        item_emb_sub = tf.reshape(item_emb_sub, [-1, hidden_units])
        din_sub = tf.concat([u_emb_sub, item_emb_sub, u_emb_sub * item_emb_sub], axis=-1)
        din_sub = tf.layers.batch_normalization(inputs=din_sub, name='b1', reuse=True)
        d_layer_1_sub = tf.layers.dense(din_sub, 80, activation=tf.nn.sigmoid, name='f1', reuse=True)
        # d_layer_1_sub = dice(d_layer_1_sub, name='dice_1_sub')
        d_layer_2_sub = tf.layers.dense(d_layer_1_sub, 40, activation=tf.nn.sigmoid, name='f2', reuse=True)
        # d_layer_2_sub = dice(d_layer_2_sub, name='dice_2_sub')
        d_layer_3_sub = tf.layers.dense(d_layer_2_sub, 1, activation=None, name='f3', reuse=True)
        d_layer_3_sub = tf.reshape(d_layer_3_sub, [-1, predict_ads_num])
        self.logits_sub = tf.sigmoid(item_b[:predict_ads_num] + d_layer_3_sub)
        self.logits_sub = tf.reshape(self.logits_sub, [-1, predict_ads_num, 1])
        # -- fcn end -------

        self.mf_auc = tf.reduce_mean(tf.to_float(x > 0))
        self.score_i = tf.sigmoid(i_b + d_layer_3_i)  # 正样本的的得分
        self.score_j = tf.sigmoid(j_b + d_layer_3_j)
        self.score_i = tf.reshape(self.score_i, [-1, 1])  # [B,1]
        self.score_j = tf.reshape(self.score_j, [-1, 1])  # [B,1]
        self.p_and_n = tf.concat([self.score_i, self.score_j], axis=-1)
        print(self.p_and_n.get_shape().as_list())

        # Step variable
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = \
            tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = \
            tf.assign(self.global_epoch_step, self.global_epoch_step + 1)

        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.y)
        )

        trainable_params = tf.trainable_variables()
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
        self.train_op = self.opt.apply_gradients(
            zip(clip_gradients, trainable_params), global_step=self.global_step)

    def train(self, sess, uij, l):  # uij 为mini_batch_dataset。 uij=(u, i, y, hist_i, sl)
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
            self.u: uij[0],  # 每个mini_batch中的用户列表
            self.i: uij[1],  # 每个mini_batch 中的正样本列表
            self.y: uij[2],
            self.hist_i: uij[3],  # 用户行为矩阵
            self.sl: uij[4],  # 每个用户对应的真实行为序列的长度
            self.lr: l,
        })
        return loss

    def eval(self, sess, uij):
        u_auc, socre_p_and_n = sess.run([self.mf_auc, self.p_and_n], feed_dict={
            self.u: uij[0],
            self.i: uij[1],
            self.j: uij[2],
            self.hist_i: uij[3],
            self.sl: uij[4],
        })
        return u_auc, socre_p_and_n

    def test(self, sess, uij):
        return sess.run(self.logits_sub, feed_dict={
            self.u: uij[0],
            self.i: uij[1],
            self.j: uij[2],
            self.hist_i: uij[3],
            self.sl: uij[4],
        })

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)


def extract_axis_1(data, ind):
    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)
    return res


def attention(queries, keys, keys_length):
    '''
        这里的输入有三个，候选广告queries，用户历史行为keys，以及Batch中每个行为的长度。
        这里为什么要输入一个keys_length呢，因为每个用户发生过的历史行为是不一样多的，但是输入的keys维度是固定的(都是历史行为最大的长度)，
        因此我们需要这个长度来计算一个mask，告诉模型哪些行为是没用的，哪些是用来计算用户兴趣分布的。

            queries:     [B, H]     即 i_emb
            keys:        [B, T, H]   即 h_emb
            keys_length: [B]         即 self.sl
            经过以下几个步骤得到用户的兴趣分布：

            1. 将queries变为和keys同样的形状B * T * H(B指batch的大小，T指用户历史行为的最大长度，H指embedding的长度)
            2. 通过三层神经网络得到queries和keys中每个key的权重，并经过softmax进行标准化
            3. 通过weighted sum得到最终用户的历史行为分布

    '''
    # 训练的权重w 为 每个商品对固定长度的商品分布进行训练，wij即是商品item_j 对商品item_i的影响权重
    # attention 机制网络的输入数据
    queries_hidden_units = queries.get_shape().as_list()[-1]  # shape [H]
    queries = tf.tile(queries, [1, tf.shape(keys)[1]])  # [B,H]   ---> T*[B*H]        将queries向量横向赋值T份，和向量一一交叉积计算，
    queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])  # T*[B*H] ---->[B,T,H]
    din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)  # 原始特征，相减特征，二阶特征B*T*4H

    """
    attention 部分真正的神经网络结构，并且只在H(embedding)这一维度做卷积
    """
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att',
                                    reuse=tf.AUTO_REUSE)  # 是否重复使用参数
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att', reuse=tf.AUTO_REUSE)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att', reuse=tf.AUTO_REUSE)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])  # [B,1,T]
    outputs = d_layer_3_all
    # Mask
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])  # keys_length的形状为[B, T]，mask的最大长度为T    # True / False
    key_masks = tf.expand_dims(key_masks, 1)  # [B, 1, T]
    # tf.ones_like 函数的目的是创建一个和输入参数（tensor）维度一样，元素都为1的张量
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)  # 创建相同类型的矩阵的目的：在不足的地方附上一个很小的值，而不是0，
    # 填充值用一个很大的赋值表示-2 ** 32 + 1，softmax计算值近似为0
    # tf.where(condition,x,y) 如果condition对应位置值为true，那么返回tensor对应位置为x的值，否则返回y的值
    outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]
    # padding操作，将每个样本序列中空缺的商品都赋值为（-2**32+1）
    # Scale
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

    # Activation
    outputs = tf.nn.softmax(
        outputs)  # [B, 1, T]                  # B * 1 * T，这里的outputs是attention计算出来的权重，也即是w    获得每个历史行为对应的权重

    # Weighted sum
    outputs = tf.matmul(outputs, keys)  # [B, 1, T] x [B,T,H] =  B * 1 * H 三维矩阵相乘，相乘发生在后两维，即 B * (( 1 * T ) * ( T * H ))
    # 此外作者还定义了attention_multi_items的函数，
    # def attention_multi_items(queries,keys,keys_length):
    # 它的逻辑和上面的attention完全一样，只不过，上面的attention函数处理的query只有一个候选广告，但这里的attention_multi_items 一次处理N个候选广告

    return outputs


def attention_multi_items(queries, keys, keys_length):
    '''
    queries:     [B, N, H] N is the number of ads
    keys:        [B, T, H]
    keys_length: [B]
    '''
    queries_hidden_units = queries.get_shape().as_list()[-1]  # 向量维度H
    queries_nums = queries.get_shape().as_list()[1]  # N
    queries = tf.tile(queries, [1, 1, tf.shape(keys)[1]])
    queries = tf.reshape(queries, [-1, queries_nums, tf.shape(keys)[1], queries_hidden_units])  # shape : [B, N, T, H]
    max_len = tf.shape(keys)[1]
    keys = tf.tile(keys, [1, queries_nums, 1])
    keys = tf.reshape(keys, [-1, queries_nums, max_len, queries_hidden_units])  # shape : [B, N, T, H]
    din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att', reuse=tf.AUTO_REUSE)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att', reuse=tf.AUTO_REUSE)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att', reuse=tf.AUTO_REUSE)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, queries_nums, 1, max_len])
    outputs = d_layer_3_all
    # Mask
    key_masks = tf.sequence_mask(keys_length, max_len)  # [B, T]
    key_masks = tf.tile(key_masks, [1, queries_nums])
    key_masks = tf.reshape(key_masks, [-1, queries_nums, 1, max_len])  # shape : [B, N, 1, T]
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    outputs = tf.where(key_masks, outputs, paddings)  # [B, N, 1, T]

    # Scale
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

    # Activation
    outputs = tf.nn.softmax(outputs)  # [B, N, 1, T]
    outputs = tf.reshape(outputs, [-1, 1, max_len])
    keys = tf.reshape(keys, [-1, max_len, queries_hidden_units])
    # print outputs.get_shape().as_list()
    # print keys.get_sahpe().as_list()
    # Weighted sum
    outputs = tf.matmul(outputs, keys)
    outputs = tf.reshape(outputs, [-1, queries_nums, queries_hidden_units])  # [B, N, 1, H]
    print(outputs.get_shape().as_list())
    return outputs


"""
B：batch_size
T: 用户历史行为序列的最大长度
H: embedding 的维度

query: 即框架图中的候选广告
facts: 用户历史行为序列，即用户过去点击的广告

attention_size：代码中没有用到，可以忽略

mask：大小为B*T,由于每个用户的历史行为序列长度不是固定的，所以用mask来进行标记，
对于每一行，第1~j（j<=T）个元素为True，第j+1~ T 个元素为False

#由于query大小只有 B*H ,所以先将其扩展为 B*T*H大小的queries后，与facts，queries-facts、queries*facts 拼接后的din_all作为
三层深度网络的输入，最后输出B*T*1大小的d_layer_3_all，再通过reshape变为B*1*T的scores


"""
