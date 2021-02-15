# -*- coding:UTF-8 -*-


import os
import sys
import time
import random
import datetime
from optparse import OptionParser
import pandas as pd
import numpy as np
import scipy.stats as st
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
import cPickle as pickle 
import math

def para_define(parser):

    parser.add_option("-l", "--log", action="store", \
                        type="string", \
                        dest="log", \
                        help="show the log", \
                        default="1")

    parser.add_option("-c", "--samples", action="store", \
                        type="string", \
                        dest="f_samples", \
                        help="f_samples", \
                        default="data/rank/rank_sample.raw.vec")

def log_time(s):
    if options.log == "1": 
        now = datetime.datetime.now()
        print("[%s][%s.%s][%s]" %(time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time())), now.second, now.microsecond, s))

def PR(a_predict, a_test):
    cnt0 = 0
    cnt0_1 = 0
    cnt_match = 0
    cnt_match_1 = 0 
    total_err = 0.0
    for (i,v) in enumerate(a_test):
        err = a_predict[i] - a_test[i]
        print( a_predict[i], a_test[i], err)
        total_err += err * err
        cnt0 += 1
        if a_predict[i] < 0.5 and a_test[i] < 0.5:
            cnt_match += 1
        elif a_predict[i] > 0.5 and a_test[i] > 0.5:
            cnt_match += 1
            cnt_match_1 += 1
        if a_test[i] > 0.5:
            cnt0_1 += 1
            
    print("rmse: %f" % math.sqrt(total_err/ len(a_test)))

    print("total:%s\tmatch:%s\tratio:%s\ttotal_1:%s\tmatch_1:%s\tratio:%s" %(cnt0, \
            cnt_match, \
            (cnt_match+0.0)/cnt0, \
            cnt0_1, \
            cnt_match_1, \
            (cnt_match_1+0.0)/cnt0_1))



if __name__ == "__main__":
    
    # op config
    parser = OptionParser()
    para_define(parser)
    (options, args) = parser.parse_args()

    log_time("loading samples...")
    infile = options.f_samples

    features = {}
    labels = {}
    for line in open(infile):
        items = line.strip('\n').split('\t')
        bid = items[1]
        features.setdefault(bid, [])
        labels.setdefault(bid, [])
        #print line
        features[bid].append(map(float, items[2:]))
        labels[bid].append(float(items[0]))

    X_train = []
    y_train = []
    X_valid = []
    y_valid = []

    X_test = []
    y_test = []
    group_train = []
    group_valid = []
    bid_test = []
    num = len(features)
    count = 0
    # 正负样本
    count2 = 0
    for bid in features:
        if '_' not in bid:
            continue
        if count2 % 5 != 0:
            X_train += features[bid]
            y_train += labels[bid]
            group_train.append(len(features[bid]))
        elif count2 % 5 == 0:
            X_test += features[bid]
            y_test += labels[bid]
            bid_test += [bid] * len(features[bid])
        count2 += 1
    
    num -= count2
    for bid in features:
        if '_' in bid:
            continue
        if count < (num / 2):
            Xtrain, Xtest, ytrain, ytest = train_test_split(features[bid], labels[bid], test_size=0.2, random_state=1)
            X_train += Xtrain
            y_train += ytrain
            X_test += Xtest
            y_test += ytest
            bid_test += [bid] * len(ytest)
            group_train.append(len(Xtrain))
        else:
            Xtrain2, Xtest2, ytrain2, ytest2 = train_test_split(features[bid], labels[bid], test_size=0.2, random_state=1)
            X_train += Xtrain2
            y_train += ytrain2
            X_valid += Xtest2
            y_valid += ytest2
            group_train.append(len(Xtrain2))
            group_valid.append(len(Xtest2))
            print("bid: %s, train group len: %d, valid group len: %d" % \
                (bid, len(Xtrain2), len(Xtest2)))
        count += 1
    '''
    for bid in features:
        Xtrain, Xtest, ytrain, ytest = train_test_split(features[bid], labels[bid], test_size=0.1, random_state=1)
        X_test += Xtest
        y_test += ytest
        bid_test += [bid] * len(ytest)
        Xtrain2, Xtest2, ytrain2, ytest2 = train_test_split(Xtrain, ytrain, test_size=0.1, random_state=1)
        X_train += Xtrain2
        y_train += ytrain2
        X_valid += Xtest2
        y_valid += ytest2
        group_train.append(len(Xtrain2))
        group_valid.append(len(Xtest2))
        print "bid: %s, train group len: %d, valid group len: %d" % \
            (bid, len(Xtrain2), len(Xtest2))
    '''

    X_train, tid_train = zip(*[(item[:len(item) - 1], item[len(item) - 1]) for item in X_train])
    X_valid, tid_valid = zip(*[(item[:len(item) - 1], item[len(item) - 1]) for item in X_valid])
    X_test, tid_test = zip(*[(item[:len(item) - 1], item[len(item) - 1]) for item in X_test])
    print(X_train[0], tid_train[0])
    print(len(X_train[0]))
    
    print("train size: %d, valid size: %d, test size: %d" % \
        (len(X_train), len(X_valid), len(X_test)))

    params = {}
    params['objective'] = 'rank:pairwise'
    # learning_rate
    params['eta'] = 0.039689061993067813
    params['max_depth'] = 19
    #params['nthread'] = -1
    params['gamma'] = 0.23719530543536704
    params['min_child_weight'] = 1.4554682097042133
    params['alpha'] = 0.22155136208398515
    params['subsample'] = 0.9457998615452512
    params['colsample_bytree'] = 0.92547437073957228
    #params['eval_metric'] = 'ndcg@10-'
    params['eval_metric'] = 'ndcg'
    # 迭代次数, n_estimators
    num_bound = 89
    '''
    params['eta'] = 0.36338702208872725
    params['max_depth'] = 5
    #params['nthread'] = -1
    params['gamma'] = 1.621738088517054
    params['min_child_weight'] = 24.568642954183382
    params['alpha'] = 7.3225327016340884
    params['subsample'] = 0.69815222586774039
    params['colsample_bytree'] = 0.9030215279067747
    params['eval_metric'] = 'ndcg'
    # 迭代次数, n_estimators
    num_bound = 160
    '''
    '''
    params['eta'] = 0.12399160024124577
    params['max_depth'] = 8
    #params['nthread'] = -1
    params['gamma'] = 7.3288153230388424
    params['min_child_weight'] = 71.987432485623742
    params['alpha'] = 2.2115777582276177
    params['subsample'] = 0.92392406755127765
    params['colsample_bytree'] = 0.96736365787885914
    params['eval_metric'] = 'ndcg'
    # 迭代次数, n_estimators
    num_bound = 147
    '''
    '''
    params['eta'] = 0.1
    params['max_depth'] = 6
    #params['nthread'] = -1
    params['gamma'] = 1.0
    params['min_child_weight'] = 0.1
    params['eval_metric'] = 'ndcg'
    # 迭代次数, n_estimators
    num_bound = 6
    '''

    dtrain = xgb.DMatrix(np.array(X_train), label=np.array(y_train))
    dtrain.set_group(group_train)
    dvalid = xgb.DMatrix(np.array(X_valid), label=np.array(y_valid))
    dvalid.set_group(group_valid)
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    bst = xgb.train(params, dtrain, num_bound, watchlist, early_stopping_rounds=50, verbose_eval=1)

    log_time('save model ...')
    bst.save_model("data/comment_xgb_rank.vec.pickle.model")
    log_time('save model done')

    log_time('predict test ...')
    dtest = xgb.DMatrix(np.array(X_test))
    ptest = bst.predict(dtest)
    print (ptest)
    dict_rank_test = {}
    for i in range(0, len(ptest)):
        dict_rank_test.setdefault(bid_test[i], [])
        dict_rank_test[bid_test[i]].append((tid_test[i], y_test[i], ptest[i]))
        print ("bid: %s, tid: %d, level: %d, predict score: %f" % (bid_test[i], tid_test[i], y_test[i], ptest[i]))

    sum_ndcg = 0
    for bid in dict_rank_test:
        dict_rank_test[bid].sort(lambda x, y: -cmp(x[1], y[1]))
        idcg = 0
        for i, (tid, r, v) in enumerate(dict_rank_test[bid]):
            if i == 0:
                idcg += r
            else:
                idcg += (r / math.log(i + 1, 2))
        dcg = 0
        dict_rank_test[bid].sort(lambda x, y: -cmp(x[2], y[2]))
        for i, (tid, r, v) in enumerate(dict_rank_test[bid]):
            if i == 0:
                dcg += r
            else:
                dcg += (r / math.log(i + 1, 2))
        sum_ndcg += dcg / idcg
        print ("bid: %s, dcg: %f, idcg: %f, ndcg: %f" % (bid, dcg, idcg, dcg / idcg))
    print("mean of ndcg: %f" % (sum_ndcg / len(dict_rank_test)))

