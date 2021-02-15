import tensorflow as tf

from Dice import dice


class Model(object):
    def __init__(self, user_count, item_count, cate_count, cate_list):
        self.u = tf.placeholder(tf.int32, [None, ], name='user')  # self.u 用户id序列
        self.i = tf.placeholder(tf.int32, [None, ], name='item')  # 正样本的item ，目标节点序列（就是正/负样本）
        self.j = tf.placeholder(tf.int32, [None, ], name='item_j')  # 负样本的item
        # 其中model中关于self.j的代码，它只会在运行测试集的时候会被用到，这和TF的静态图特性有关
        self.y = tf.placeholder(tf.float32, [None, ], name='label')  # 目标节点对应的label序列，正样本对应1，负样本对应0

        # 用户行为特征（user Behavior）中的item序列，
        self.hist_i = tf.placeholder(tf.int32, [None, None], name='history_i')
        # user behavior 中序列的真实序列长度
        self.sl = tf.placeholder(tf.int32, [None, ], name='sequence_length')  # 记录用户行为序列的真实长度
        self.lr = tf.placeholder(tf.float64, name='learning_rate')

        hidden_units = 32
        # user_id 的embedding weight , user_count 是user_id的hash bucket size
        user_emb_w = tf.get_variable("user_emb_w", [user_count, hidden_units])

        # item_id的embedding weight ,item_count是item_id 的hash bucket size
        item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_units // 2])  # 所有商品的embedding

        item_b = tf.get_variable("item_b", [item_count], initializer=tf.constant_initializer(0.0))

        cate_emb_w = tf.get_variable("cate_emb_w", [cate_count, hidden_units // 2])  # 所有类目的embedding
        cate_list = tf.convert_to_tensor(cate_list, dtype=tf.int64)

        # u_emb = tf.nn.embedding_lookup(user_emb_w,self.u)

        # ic是item到category的转换 ,从cate_list中取出正样本的cate
        self.ic = tf.gather(cate_list, self.i)

        # 正样本的embedding，正样本包括item 和 cate
        i_emb = tf.concat(values=[
            tf.nn.embedding_lookup(item_emb_w, self.i),
            tf.nn.embedding_lookup(cate_emb_w, self.ic)
        ], axis=1)

        i_b = tf.gather(item_b, self.i)

        self.jc = tf.gather(cate_list, self.j)  # 从cate_list中取出负样本的cate
        j_emb = tf.concat([
            tf.nn.embedding_lookup(item_emb_w, self.j),
            tf.nn.embedding_lookup(cate_emb_w, self.jc),
        ], axis=1)  # 负样本的embedding， 负样本包括item 和cate
        j_b = tf.gather(item_b, self.j)

        self.hc = tf.gather(cate_list, self.hist_i)  # 用户行为序列（user behavior）中的cate 序列

        h_emb = tf.concat([
            tf.nn.embedding_lookup(item_emb_w, self.hist_i),
            tf.nn.embedding_lookup(cate_emb_w, self.hc),
        ], axis=2)  # 用户行为序列（user behavior）的embedding，包括item序列和cate序列

        hist = attention(i_emb, h_emb, self.sl)  # attention 操作

        hist = tf.layers.batch_normalization(inputs=hist)
        hist = tf.reshape(hist, [-1, hidden_units])
        hist = tf.layers.dense(hist, hidden_units)

        u_emb = hist
        """
        上面的代码主要对用户embedding layer,商品的embedding layer 以及类目的embedding layer 进行初始化，然后获取一个batch中目标
        节点对应的embeddding，保存在i_emb中，它由商品（Goods）和类目（Cate）embedding进行concatenation，
        后面对self.hist_i进行处理，其保存了用户的历史行为序列，大小为[B,T],所以进行embedding_lookup时，输出大小为[B,T,H/2];之后将Goods和
        Cate的embedding进行concat
        
        """

        # fcn begin
        din_i = tf.concat([u_emb, i_emb], axis=-1)  # u_emb表示用户兴趣，i_emb 候选广告对应的embedding
        # u_emb * i_emb 用户兴趣和候选广告的交叉特征
        din_i = tf.layers.batch_normalization(inputs=din_i, name='b1')
        d_layer_1_i = tf.layers.dense(din_i, 80, activation=None, name='f1')
        d_layer_1_i = dice(d_layer_1_i, name='dice_1_i')
        d_layer_2_i = tf.layers.dense(d_layer_1_i, 40, activation=None, name='f2')
        d_layer_2_i = dice(d_layer_2_i, name='dice_2_i')
        d_layer_3_i = tf.layers.dense(d_layer_2_i, 1, activation=None, name='f3')

        din_j = tf.concat([u_emb, j_emb], axis=-1)
        din_j = tf.layers.batch_normalization(inputs=din_j, name='b1', reuse=True)
        d_layer_1_j = tf.layers.dense(din_j, 80, activation=None, name='f1', reuse=True)
        d_layer_1_j = dice(d_layer_1_j, name='dice_1_j')
        d_layer_2_j = tf.layers.dense(d_layer_1_j, 40, activation=None, name='f2', reuse=True)
        d_layer_2_j = dice(d_layer_2_j, name='dice_2_j')
        d_layer_3_j = tf.layers.dense(d_layer_2_j, 1, activation=None, name='f3', reuse=True)

        d_layer_3_i = tf.reshape(d_layer_3_i, [-1])
        d_layer_3_j = tf.reshape(d_layer_3_j, [-1])

        x = i_b - j_b + d_layer_3_i - d_layer_3_j  # [B]
        self.logits = i_b + d_layer_3_i

        # logits for all item:
        u_emb_all = tf.expand_dims(u_emb, 1)
        u_emb_all = tf.tile(u_emb_all, [1, item_count, 1])

        all_emb = tf.concat([
            item_emb_w,
            tf.nn.embedding_lookup(cate_emb_w, cate_list)
        ], axis=1)
        all_emb = tf.expand_dims(all_emb, 0)
        all_emb = tf.tile(all_emb, [512, 1, 1])
        din_all = tf.concat([u_emb_all, all_emb], axis=-1)
        din_all = tf.layers.batch_normalization(inputs=din_all, name='b1', reuse=True)
        d_layer_1_all = tf.layers.dense(din_all, 80, activation=None, name='f1', reuse=True)
        d_layer_1_all = dice(d_layer_1_all, name='dice_1_all')
        d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=None, name='f2', reuse=True)
        d_layer_2_all = dice(d_layer_2_all, name='dice_2_all')
        d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3', reuse=True)
        d_layer_3_all = tf.reshape(d_layer_3_all, [-1, item_count])

        self.logits_all = tf.sigmoid(item_b + d_layer_3_all)
        # -- fcn end -------

        self.mf_auc = tf.reduce_mean(tf.to_float(x > 0))
        self.score_i = tf.sigmoid(i_b + d_layer_3_i)
        self.score_j = tf.sigmoid(j_b + d_layer_3_j)
        self.score_i = tf.reshape(self.score_i, [-1, 1])
        self.score_j = tf.reshape(self.score_j, [-1, 1])
        self.p_and_n = tf.concat([self.score_i, self.score_j], axis=-1)

        # Step variable
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = \
            tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = \
            tf.assign(self.global_epoch_step, self.global_epoch_step + 1)

        # loss and train
        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.y)
        )

        trainable_params = tf.trainable_variables()
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.loss)

    def train(self, sess, uij, l):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
            # self.u : uij[0],
            self.i: uij[1],
            self.y: uij[2],
            self.hist_i: uij[3],
            self.sl: uij[4],
            self.lr: l
        })

        return loss

    def eval(self, sess, uij):
        u_auc, socre_p_and_n = sess.run([self.mf_auc, self.p_and_n], feed_dict={
            # self.u: uij[0],
            self.i: uij[1],
            self.j: uij[2],
            self.hist_i: uij[3],
            self.sl: uij[4],
        })
        return u_auc, socre_p_and_n

    def test(self, sess, uid, hist_i, sl):
        return sess.run(self.logits_all, feed_dict={
            self.u: uid,
            self.hist_i: hist_i,
            self.sl: sl,
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


# Activation Unit
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

    queries_hidden_units = queries.get_shape().as_list()[-1]  # shape [H]

    queries = tf.tile(queries, [1, tf.shape(keys)[1]])  # [B,H]   ---> T*[B*H]

    queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])  # T*[B*H] ---->[B,T,H]

    din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)  # attention操作，输出维度 B*T*4H
    # 三层全链接
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att')  # [B,T,80]
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att')  # [B,T,40]
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att')  # B*T*1

    outputs = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])  # B*1*T

    # Mask
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])  # [B,T]
    key_masks = tf.expand_dims(key_masks, 1)  # B*1*T
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)  # 在补足的地方附上一个很小的值，而不是0，
    # 填充值用一个很大的赋值表示-2 ** 32 + 1，softmax计算值近似为0
    outputs = tf.where(key_masks, outputs, paddings)  # B * 1 * T
    # padding操作，将每个样本序列中空缺的商品都赋值为（-2**32+1）

    # Scale
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

    # Activation
    outputs = tf.nn.softmax(outputs)  # B * 1 * T，这里的outputs是attention计算出来的权重，也即是w    获得每个历史行为对应的权重

    # Weighted Sum
    outputs = tf.matmul(outputs, keys)  # B * 1 * H 三维矩阵相乘，相乘发生在后两维，即 B * (( 1 * T ) * ( T * H ))

    # 此外作者还定义了attention_multi_items的函数，
    # def attention_multi_items(queries,keys,keys_length):
    # 它的逻辑和上面的attention完全一样，只不过，上面的attention函数处理的query只有一个候选广告，但这里的attention_multi_items 一次处理N个候选广告

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

attention机制：利用了用户兴趣多样性以及当前候选商品仅与用户一部分兴趣有关这一特点，引入注意力机制
GAUC：由不不同用户之间天生具有差异：有些用户天生点击率高，针对这一点对传统AUC进行改进。消除用户偏差对模型对影响
自适应对正则化方法：根据特征出现的频率，给予不同对正则化强度
Dice激活函数：节点是否被激活，取决于该节点数据的分布
"""
