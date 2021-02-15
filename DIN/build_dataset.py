import random
import pickle
"""
用用户的行为序列数据，使用前k个看过的商品预测第k+1个物品，training dataset is generated with k=1,2,...n-2 for
each user, 测试集是使用前面 n-1 个看过的物品预测最后一个物品
"""
random.seed(1024)
"""
Pickle 每次序列化生成的字符串有独立头尾，pickle.load() 只会读取一个完整的结果，
所以你只需要在 load 一次之后再 load 一次，就能读到第二次序列化的 [‘asd’, (‘ss’, ‘dd’)]。
如果不知道文件里有多少 pickle 对象，可以在 while 循环中反复 load 文件对象，直到抛出异常为止。
"""

# reviewerID: user_id
# asin :item_id

with open('data/remap.pkl', 'rb') as f:
    reviews_df = pickle.load(f)
    cate_list = pickle.load(f)
    user_count, item_count, cate_count, example_count = pickle.load(f)

train_set = []    # 训练集
test_set = []     # 测试集
for reviewerID, hist in reviews_df.groupby('reviewerID'):  # 按照user_id进行聚合，划分成多个sub_dataframe,每个dataframe 的name 为user_id
    pos_list = hist['asin'].tolist()  # sub_group中对应的item_id列即为用户点击过的商品序列。 作为用户的正样本序列集合
    def gen_neg():
        neg = pos_list[0]     # 先初始化负样本
        while neg in pos_list:  # 判断是否在正样本集合中
            neg = random.randint(0, item_count - 1)               # 产生负样本，从全部候选商品集合中随机选则用户没有访问过的商品
        return neg

    neg_list = [gen_neg() for i in range(len(pos_list))]          # 根据用户行为序列的长度，产生相应个数的负样本

    for i in range(1, len(pos_list)):                             # 训练集，用户行为数据，采用前k个序列预测第k+1个序列生成一条样本 ，会产生n-1条样本
        hist = pos_list[:i]                                       # 前面的hist是dataframe,这里的hist是具体用户的商品浏览行为
        if i != len(pos_list) - 1:
            train_set.append((reviewerID, hist, pos_list[i], 1))  # 第i个行为作为label
            train_set.append((reviewerID, hist, neg_list[i], 0))  # 取负样本
        else:                                                     # 将预测最后一个商品行为的样本作为测试集
            label = (pos_list[i], neg_list[i])
            test_set.append((reviewerID, hist, label))            # 用户，长短行为，label（正/负）

# shuffle
random.shuffle(train_set)
random.shuffle(test_set)

# print(train_set[:2])
# print(test_set[:5])

# assert (断言) 用于判断一个表达式，在表达式条件为false的时候触发异常，可以在条件不满足程序运行的情况下直接返回错误
assert len(test_set) == user_count
# assert(len(test_set) + len(train_set) // 2 == reviews_df.shape[0])

with open('dataset.pkl', 'wb') as f:
    pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)    # user_id,history,label,flag
    pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)     # user_id,history,label
    pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)    # cate_list
    pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)  # 用户数量，商品数量，类目数

"""

将hist的asin列作为每个reviewerID(也就是用户)的正样本列表（pos_list）,注意这里的asin存的已经不是原始的item_id了，
而是通过asin_map转换过来的index。负样本列表(neg_list)为在item_count范围内产生不在pos_list中的随机数列表


例如对于reviewerID=0的用户，他的pos_list为[13179, 17993, 28326, 29247, 62275], 生成的训练集格式为（reviewerID, hist, pos_item, 1),
 (reviewerID, hist, neg_item, 0),这里需要注意hist并不包含pos_item, hist只包含在pos_item之前点击过的item，因为DIN采用类似attention的机制，
 只有历史的行为的attention才对后续的有影响，所以hist只包含pos_item之前点击的item才有意义。
"""
