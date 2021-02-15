import pickle
import pandas as pd


def to_df(file_path):
    with open(file_path, 'r') as fin:
        df = {}
        i = 0
        for line in fin:
            df[i] = eval(line)
            i += 1
        df = pd.DataFrame.from_dict(df, orient='index')   # 将dict转换为dataframe对象
        return df


reviews_df = to_df('../data/reviews_Electronics_5.json')    # 转化为dataframe格式  列分别为reviewID,asin,reviewerNamed等
with open('../data/reviews.pkl', 'wb') as f:
    pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)

meta_df = to_df('../data/meta_Electronics.json')            # 转化为dataframe格式
meta_df = meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())]    # 并且只保留reviewes文件中出现过的商品，去重
meta_df = meta_df.reset_index(drop=True)
with open('../data/meta.pkl', 'wb') as f:
    pickle.dump(meta_df, f, pickle.HIGHEST_PROTOCOL)      # 将转换完的格式保存为pkl格式



"""
reviews主要是用户买了相关商品产生的上下文信息，包括商品id,时间，评论等；
meta：关于商品本省的信息，包括商品id，名称，类别，买了还是没买
"""