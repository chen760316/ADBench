import nibabel as nib
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import average_precision_score

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('../datasets/real_outlier/optdigits.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = CatBoostClassifier()
model.fit(X_train, y_train)  # fit

# 确保 X_test 是 NumPy 数组
y_test_pred = model.predict(X_test)  # predict

"""Accuracy指标"""
print("*" * 100)
print("CatB在测试集中的分类准确度：" + str(accuracy_score(y_test, y_test_pred)))

"""Precision/Recall/F1指标"""
print("*" * 100)
# average='micro': 全局计算 F1 分数，适用于处理类别不平衡的情况。
# average='macro': 类别 F1 分数的简单平均，适用于需要均衡考虑每个类别的情况。
# average='weighted': 加权 F1 分数，适用于类别不平衡的情况，考虑了每个类别的样本量。
# average=None: 返回每个类别的 F1 分数，适用于详细分析每个类别的表现。
print("CatB在测试集中的分类精确度：" + str(precision_score(y_test, y_test_pred, average='weighted')))
print("CatB在测试集中的分类召回率：" + str(recall_score(y_test, y_test_pred, average='weighted')))
print("CatB在测试集中的分类F1分数：" + str(f1_score(y_test, y_test_pred, average='weighted')))

"""ROC-AUC指标"""
print("*" * 100)
roc_auc_test = roc_auc_score(y_test, y_test_pred, multi_class='ovr')  # 一对多方式
print("CatB在测试集中的ROC-AUC分数：" + str(roc_auc_test))

"""PR AUC指标"""
# 只要能确保y_scores值越大取正值的概率越大即可

print("*" * 100)
# 预测概率
y_scores = model.predict_proba(X_test)[:, 1]  # 返回正例概率

# # 线性缩放到 [0, 1] 范围
# min_score = np.min(y_scores)
# max_score = np.max(y_scores)
# scaled_scores = (y_scores - min_score) / (max_score - min_score)  # 归一化到 [0, 1]
# # 如果希望在较小的得分上更快接近 0，可以使用 Sigmoid 函数
# # 你可以通过调整缩放因子来控制形状
# smoothing_factor = 10  # 可以调整这个值
# probabilities = 1 / (1 + np.exp(-smoothing_factor * (scaled_scores - 0.5)))

# 计算 Precision 和 Recall
precision, recall, _ = precision_recall_curve(y_test, y_scores)
# 计算 PR AUC
pr_auc = auc(recall, precision)
print("PR AUC 分数:", pr_auc)

"""AP指标"""
# 只要能确保y_scores值越大取正值的概率越大即可

print("*" * 100)
# 预测概率
y_scores = model.predict_proba(X_test)[:, 1]  # 返回正例概率

# # 线性缩放到 [0, 1] 范围
# min_score = np.min(y_scores)
# max_score = np.max(y_scores)
# scaled_scores = (y_scores - min_score) / (max_score - min_score)  # 归一化到 [0, 1]
# # 如果希望在较小的得分上更快接近 0，可以使用 Sigmoid 函数
# # 你可以通过调整缩放因子来控制形状
# smoothing_factor = 10  # 可以调整这个值
# probabilities = 1 / (1 + np.exp(-smoothing_factor * (scaled_scores - 0.5)))
# 计算 Average Precision

ap_score = average_precision_score(y_test, y_scores)
print("AP分数:", ap_score)