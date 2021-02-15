import random
import pickle
import numpy as np

random.seed(1234)

with open('../data/reviews.pkl', 'rb') as f:
    reviews_df = pickle.load(f)
    reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]  # 将reviews_df 只保留reviewerID，asin，unixReviewTime三列
with open('../data/meta.pkl', 'rb') as f:
    meta_df = pickle.load(f)
    meta_df = meta_df[['asin', 'categories']]  # asin 即item_id  保留 item_id，cate列
    meta_df['categories'] = meta_df['categories'].map(lambda x: x[-1][-1])


def build_map(df, col_name):
    key = sorted(df[col_name].unique().tolist())  # 列元素的取值集合
    m = dict(zip(key, range(len(key))))  # key为原始信息，value为对应的index
    df[col_name] = df[col_name].map(lambda x: m[x])  # df 中列元素用对应的index替换
    return m, key


asin_map, asin_key = build_map(meta_df, 'asin')  # item_id
cate_map, cate_key = build_map(meta_df, 'categories')  # cate_id
revi_map, revi_key = build_map(reviews_df, 'reviewerID')  # user_id

user_count = len(revi_map)         # 用户数量
item_count = len(asin_map)         # 商品数量
cate_count = len(cate_map)         # 商品对应的类目数量
example_count = reviews_df.shape[0]   # 用户产生的商品行为总数

print('user_count: %d\titem_count: %d\tcate_count: %d\texample_count: %d' %
      (user_count, item_count, cate_count, example_count))

# print(meta_df.head(10))     # 查看商品的item_id.index 和 cate.index

meta_df = meta_df.sort_values('asin')           # 商品信息的dataframe 按照商品（index）进行排序
meta_df = meta_df.reset_index(drop=True)        # 去除默认添加的index列

reviews_df['asin'] = reviews_df['asin'].map(lambda x: asin_map[x])    # 将用户行为的dataframe 中的item_id 用index替换
reviews_df = reviews_df.sort_values(['reviewerID', 'unixReviewTime'])    # 按照user_id 和行为时间排序，从而达到聚合和排序
reviews_df = reviews_df.reset_index(drop=True)      # 去除默认的 index
# print(reviews_df.head(30))

reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]    # 取需要的列

cate_list = [meta_df['categories'][i] for i in range(len(asin_map))]  # 类目列表的排序是按照商品的index的大小顺序存放的
cate_list = np.array(cate_list, dtype=np.int32)              # 转化为ndarray格式

with open('../data/remap.pkl', 'wb') as f:
    pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)  # 序列化对象，file
    pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)   # 类目列表
    pickle.dump((user_count, item_count, cate_count, example_count),
                f, pickle.HIGHEST_PROTOCOL)                   # 各个特征的数量以及样本总数
    pickle.dump((asin_key, cate_key, revi_key), f, pickle.HIGHEST_PROTOCOL)     # 真实的特征集合列表
