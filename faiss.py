# faiss是为稠密向量提供高效相似度搜索和聚类的框架，一下是官网提供的demo
# 1. 首先构建训练数据和测试数据
import numpy as np
d=64                      # dimension
nb=100000                 # database size 
nq=10000                  # nb of queries
np.random.seed(1024)         # make reproduciable
xb=np.random.random(nb,d).astype("float32")           #[10000,64]的训练数据，shape为[10000,63]的查询数据
xb[:,0]+=np.arange(nb)/1000.
xq=np.random.random((nq,d)).astype('float32')
xq[:,0]+= np.arange(nq)/1000.

# 2.创建索引index，faiss创建索引对向量预处理，提高查询效率，faiss提供了多种索引的方法，这里选择最简单的暴力检索L2距离的索引：indexFlatL2
# 创建索引时必须指定向量的维度d，并且大部分索引需要训练的步骤，indexFlat2跳过这一步。
import  faiss                
index=faiss.IndexFlat2(d)        #  build the index
print(index.is_trained)

# 3. 当索引创建好并训练(如果需要)之后，就可以执行add和search方法了。add方法一般添加训练时的样本，search就是寻找相似向量
index.add(xb)      # ad vectors to the index
print(index.ntotal)

# 传入搜索向量查找相似向量

k=4           # we want to seee 4 nearest neighbor
D,I=index.search(xq,k)      # actual  search, 表示每个查询向量的最近4个向量，D表示与相似向量的距离（distance）维度， I表示相似用户的ID

print(I[:5])              # neighbors of the 5 first queries
print(D[-5:])             # neighbors of the 5 last queries

### 加速搜索
"""
如果需要存储的向量太多，通过暴力搜索索引IndexFlat2速度很慢，加速搜索的方法的索引IndexVFFlat(倒排文件)。其实是使用K-means建立聚类中心，
然后通过查询最近的聚类中心，然后比较聚类中的所有向量得到相似的向量
创建indexVFFlat时需要指定一个其他的索引作为量化器（quantizer）来计算距离或者相似度。add方法之前需要先训练
参数简介：
faiss.METRIC_L2: faiss定义了两种衡量相似度的方法(metrics)，分别为faiss.METRIC_INNER_PRODUCT。一个是欧式距离，一个向量内积
nlist：聚类中心的个数
k：查找最相似的k个向量
index.nprobe:查找聚类中心的个数，默认为1个
"""
nlist=100           # 聚类中心的个数
k=4
quantizer=faiss.IndexFlatL2(d)   # the other index
index=faiss.indexIVFFlat(quantizer,d,nlist,faiss.METRIC_L2)     # here we specify METRIC_L2,by default it performs inner-product search
assert not  index.is_trained
index.train(xb)
assert index.is_trained

index.add(xb)            # add may be a bit slower as well
D,I= index.search(xq,k)               # actual search
print(I[-5:])                         # neighbors of the 5 last queries

index.nporbe=10                       # default nprobe is 1, try a few more
D,I=index.search(xq,k)
print(I[-5:])


##减少内存的方法：使用 磁盘存储inverted indexes 的方式
"""
上面我们看到的索引indexFlatL2和indexIVFFlat都会全量存储所有的向量在内存中，为满足大的数据量的需求，faiss提供一种基于Product Quantizer(乘积量化)的压缩算法编码
向量大小到指定的字节数。此时，储存的向量压缩过的，查询的距离也是近似的。

"""
nlist=100
m=8
k=4
quantizer=faiss.IndexFlatL2(d)   # this remains the same

index=faiss.IndexIVFPQ(quantizer,d,nlist,m,8)   # 8 specifies that each sub_vector is encoded as 8 bits

index.train(xb)
index.add(xb)
D,I=index.search(xb[:5],k)            # sanity check
print(I)
print(D)
index.nprobe=10                       # make comparable with experiment above
D,I=index.search(xq,k)                
print(I[-5:])


## 简化索引的表达
"""
通过上面indexIVFFlat和indexIVVFPQ我们可以看到，他们的构造需要先提供另外一个index。类似的，faiss还提供pca，lsh等方法，有时候他们会组合使用，
这样组合的对构造索引会比较麻烦，faiss提供了通过字符串表达的方式构造索引。
如，下面表达式就能表示上面的创建indexIVFPQ的实例

"""
index=faiss.index_factory(d,"IVF100,PQ8")


# demo: https://github.com/facebookresearch/faiss/tree/master/demos
# 每类索引的简写可查询:https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
# https://blog.csdn.net/weixin_42285429/article/details/108693326?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_baidulandingword-10&spm=1001.2101.3001.4242
# 框架详解
# https://blog.csdn.net/xiaoxu2050/article/details/84982478?utm_medium=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromBaidu-1.not_use_machine_learn_pai&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromBaidu-1.not_use_machine_learn_pai





































