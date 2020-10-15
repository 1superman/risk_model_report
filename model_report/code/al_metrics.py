# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:57:36 2019

@author: chengjinkang
"""

from sklearn.metrics import *
from sklearn.linear_model import LinearRegression
from al_data_tool import *
import numpy as np
import pandas as pd
import math

def to_score(x):
    import math
    if x <=0.001:
        x =0.001
    elif x >=0.999:
        x =0.999
        
    A = 404.65547022
    B = 72.1347520444
    result = int(round(A-B*math.log(x/(1-x))))
    
    if result < 300:
        result=300
    if result>900:
        result=900
    return result

# 单列woe值
def cal_woe(Xvar, Yvar):
    """
    :param Xvar: 分箱列
    :param Yvar: y_label列
    :return: dataframe
    """

    def woe(df):
        d = pd.Series(df['y'].value_counts(),index=[0,1]).fillna(0)
        bin_good = d[0] if d[0] > 0 else d[1] * 0.1
        bin_bad = d[1] if d[1] > 0 else d[0] * 0.1
        woe = math.log((bin_bad / total_bad) * (total_good / bin_good))
        iv = (bin_bad/total_bad - bin_good/total_good)*woe

        return  pd.Series({'woe': woe,'iv':iv})

    y_value_counts = Yvar.value_counts()
    total_good = y_value_counts[0]
    total_bad = y_value_counts[1]

    df_data=pd.DataFrame({'bin':Xvar, 'y':Yvar})
    df_result = df_data.groupby('bin').apply(lambda df:woe(df)).reset_index()
    df_result['feature'] = Xvar.name

    return df_result



# 单列iv值
def cal_iv(Xvar, Yvar):
    """
    :param Xvar: 分箱列
    :param Yvar: y_label列
    :return: list
    """

    return cal_woe(Xvar, Yvar)['iv'].sum()


# 批量计算woe值
def cal_woe_batch(df, X, y, n_jobs=10):
    """
    :param df: 分完箱的dataframe
    :param X: 需要计算woe值的列名
    :param y: y的列名
    :param n_jobs: 最大的并行任务数
    :return: dataframe
    """
    woe = Parallel(n_jobs=n_jobs)(delayed(cal_woe)(df[col], df[y]) for col in X)
    woe = pd.concat(woe)
    return woe


# 批量计算iv值
def cal_iv_batch(df, X, y, n_jobs=10):
    """
    :param df: 分完箱的dataframe
    :param X: 需要计算iv值的列名
    :param y: y的列名
    :param n_jobs: 最大的并行任务数
    :return: dataframe
    """
    iv_list = Parallel(n_jobs=n_jobs)(delayed(cal_iv)(df[col], df[y]) for col in X)
    names = X
    df_iv = pd.DataFrame({'feature': names, 'iv': iv_list})

    return df_iv


# 等距分箱的iv值
def get_df_iv_equidistance(df_src, X, y, bins, n_jobs):

    df = cut_batch(df_src, X, bins, n_jobs)

    return cal_iv_batch(df, X, y, n_jobs)


# 等频分箱的iv值
def get_df_iv_equifrequency(df_src, X, y, bins, n_jobs):

    df = qcut_batch(df_src, X, bins, n_jobs)

    return cal_iv_batch(df, X, y, n_jobs)


# 等距分箱的woe值
def get_df_woe_equidistance(df_src, X, y, bins, n_jobs):
    df = cut_batch(df_src, X, bins, n_jobs)

    return cal_woe_batch(df, X, y, n_jobs)


# 等频分箱的woe值
def get_df_woe_equifrequency(df_src, X, y, bins, n_jobs):
    df = qcut_batch(df_src, X, bins, n_jobs)

    return cal_woe_batch(df, X, y, n_jobs)


# 卡方分箱的woe值
def get_df_woe_chi_merge(df, X, flag, confidenceVal=3.841, bin=10, sample_rate=0.03, sample=None, n_jobs=10):
    """
    :param df: 传入一个数据集包含需要卡方分箱的变量与正负样本标识（正样本为1，负样本为0）
    :param X: 需要卡方分箱的变量（数组）
    :param flag: 正负样本标识的名称（字符串）
    :param confidenceVal: 置信度水平（默认是不进行抽样95%）
    :param bin: 最多箱的数目
    :param sample_rate: 若某一组里的样本数量比例小于该值，进行合并
    :param sample: 为抽样的数目（默认是不进行抽样），因为如果观测值过多运行会较慢
    :param n_jobs: 最大的并行任务数
    :return:dataframe
    """
    df = chi_merge_cut_batch(df, X, flag, confidenceVal, bin, sample_rate, sample, n_jobs)

    return cal_woe_batch(df, X, flag, n_jobs=n_jobs)


# 卡方分箱的iv值
def get_df_iv_chi_merge(df, X, flag, confidenceVal=3.841, bin=10, sample_rate=0.03, sample=None, n_jobs=10):
    """
    :param df: 传入一个数据集包含需要卡方分箱的变量与正负样本标识（正样本为1，负样本为0）
    :param X: 需要卡方分箱的变量（数组）
    :param flag: 正负样本标识的名称（字符串）
    :param confidenceVal: 置信度水平（默认是不进行抽样95%）
    :param bin: 最多箱的数目
    :param sample_rate: 若某一组里的样本数量比例小于该值，进行合并
    :param sample: 为抽样的数目（默认是不进行抽样），因为如果观测值过多运行会较慢
    :param n_jobs: 最大的并行任务数
    :return:dataframe
    """
    df = chi_merge_cut_batch(df, X, flag, confidenceVal, bin, sample_rate, sample, n_jobs)

    return cal_iv_batch(df, X, flag, n_jobs)


# auc
def get_roc_auc_score(y_true, y_pred):
    if y_true.nunique() != 2:
        return np.nan
    else:
        return roc_auc_score(y_true, y_pred)


# ks
def get_ks(y_true, y_pred):
    fpr, tpr, thre = roc_curve(y_true, y_pred, pos_label=1)
    return abs(fpr - tpr).max()

def get_classifier_ks(y_true, y_pred):
    fpr, tpr, thre = roc_curve(y_true, y_pred, pos_label=1)
    return abs(fpr - tpr).max()

# corr
def get_corr(df):
    return df.corr()


# accuracy
def get_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


# precision
def get_precision(y_true, y_pred):
    if len(np.unique(y_true)) == 2:
        return precision_score(y_true, y_pred)
    else:
        return precision_score(y_true, y_pred, average='macro')


# recall
def get_recall(y_true, y_pred):
    if len(np.unique(y_true)) == 2:
        return recall_score(y_true, y_pred)
    else:
        return precision_score(y_true, y_pred, average='macro')


# f1
def get_f1(y_true, y_pred):
    if len(np.unique(y_true)) == 2:
        return f1_score(y_true, y_pred)
    else:
        return f1_score(y_true, y_pred, average='macro')

def get_vif(X:pd.DataFrame, y:pd.Series):
    model = LinearRegression()
    model.fit(X,y)
    y_pred = model.predict(X)
    r2 = r2_score(y,y_pred)
    vif = 1 / (1 - r2)
    return vif










