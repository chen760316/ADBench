"""
无监督离群值检测算法修复效果测试
"""
import re

import pandas as pd
import  os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from lime.lime_tabular import LimeTabularExplainer
from tensorflow.python.ops.metrics_impl import accuracy

from adbench.baseline.DAGMM.run import DAGMM


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
np.set_printoptions(threshold=np.inf)



def run(path):
    data = pd.read_csv(path)
    # 如果数据量超过20000行，就随机采样到20000行
    if len(data) > 20000:
        data= data.sample(n=20000, random_state=42)

    enc = LabelEncoder()
    label_name = data.columns[-1]

    # 原始数据集D对应的Dataframe
    data[label_name] = enc.fit_transform(data[label_name])

    # 检测非数值列
    non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns

    # 为每个非数值列创建一个 LabelEncoder 实例
    encoders = {}
    for column in non_numeric_columns:
        encoder = LabelEncoder()
        data[column] = encoder.fit_transform(data[column])
        encoders[column] = encoder  # 保存每个列的编码器，以便将来可能需要解码
    X = data.values[:, :-1]
    y = data.values[:, -1]

    # 找到分类特征的列名
    categorical_columns = data.select_dtypes(exclude=['float']).columns[:-1]
    # 获取分类特征对应的索引
    categorical_features = [data.columns.get_loc(col) for col in categorical_columns]

    # 统计不同值及其数量
    unique_values, counts = np.unique(y, return_counts=True)

    # 输出结果
    # for value, count in zip(unique_values, counts):
    #     print(f"标签: {value}, 数量: {count}")

    # 找到最小标签的数量
    min_count = counts.min()
    total_count = counts.sum()

    # 计算比例
    proportion = min_count / total_count
    #print(f"较少标签占据的比例: {proportion:.4f}")
    min_count_index = np.argmin(counts)  # 找到最小数量的索引
    min_label = unique_values[min_count_index]  # 对应的标签值

    # section 数据特征缩放

    # 对不同维度进行标准化
    X = StandardScaler().fit_transform(X)
    # 记录原始索引
    original_indices = np.arange(len(X))
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y, original_indices, test_size=0.3, random_state=1)
    # 加入随机噪声的比例
    noise_level = 0.2
    # 计算噪声数量
    n_samples = X.shape[0]
    n_noise = int(noise_level * n_samples)
    # 随机选择要添加噪声的样本
    noise_indices = np.random.choice(n_samples, n_noise, replace=False)
    # 添加高斯噪声到特征
    X_copy = np.copy(X)
    X_copy[noise_indices] += np.random.normal(0, 1, (n_noise, X.shape[1]))
    # 从含噪数据中生成训练数据和测试数据
    X_train_copy = X_copy[train_indices]
    X_test_copy = X_copy[test_indices]
    feature_names = data.columns.values.tolist()
    combined_array = np.hstack((X_copy, y.reshape(-1, 1)))  # 将 y 重新调整为列向量并合并
    # 添加噪声后的数据集D'对应的Dataframe
    data_copy = pd.DataFrame(combined_array, columns=feature_names)
    # 训练集中添加了高斯噪声的样本在原始数据集D中的索引
    train_noise = np.intersect1d(train_indices, noise_indices)
    # 测试集中添加了高斯噪声的样本在原始数据集D中的索引
    test_noise = np.intersect1d(test_indices, noise_indices)
    # print("训练集中的噪声样本为：", train_noise)
    # print("测试集中的噪声样本为：", test_noise)

    # SECTION M𝑜 (𝑡, D) 针对元组异常的无监督异常检测器GOAD
    epochs = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_trans = 64
    random_state = 42

    # choice DAGMM异常检测器
    out_clf = DAGMM(seed=random_state)
    out_clf.fit(X_train,y_train=np.repeat(0, X_train.shape[0]))
    out_clf_noise = DAGMM(seed=random_state)
    out_clf_noise.fit(X_train_copy,y_train=np.repeat(0, X_train_copy.shape[0]))


    # SECTION 借助异常检测器，在训练集上进行异常值检测。
    #  经过检验，加入高斯噪声会影响异常值判别

    # subsection 从原始训练集中检测出异常值索引

    #print("*"*100)
    #train_scores = out_clf.decision_function(X_train)
    train_pred_labels = out_clf.predict_label(X_train,X_train)
    #print("训练集中异常值判定阈值为：", out_clf.threshold_)
    train_outliers_index = []
    #print("训练集样本数：", len(X_train))
    for i in range(len(X_train)):
        if train_pred_labels[i] == 1:
            train_outliers_index.append(i)
    # 训练样本中的异常值索引
    # print("训练集中异常值索引：", train_outliers_index)
    # print("训练集中的异常值数量：", len(train_outliers_index))
    # print("训练集中的异常值比例：", len(train_outliers_index)/len(X_train))


    # subsection 从原始测试集中检测出异常值索引
    #test_scores = out_clf.decision_function(X_test)
    test_pred_labels = out_clf.predict_label(X_train, X_test)
    #print("测试集中异常值判定阈值为：", out_clf.threshold_)
    test_outliers_index = []
    #print("测试集样本数：", len(X_test))
    for i in range(len(X_test)):
        if test_pred_labels[i] == 1:
            test_outliers_index.append(i)
    # 训练样本中的异常值索引
    # print("测试集中异常值索引：", test_outliers_index)
    # print("测试集中的异常值数量：", len(test_outliers_index))
    # print("测试集中的异常值比例：", len(test_outliers_index)/len(X_test))

    # section 从加噪数据集的训练集和测试集中检测出的异常值

    # subsection 从加噪训练集中检测出异常值索引
    #train_scores_noise = out_clf_noise.decision_function(X_train_copy)
    train_pred_labels_noise = out_clf_noise.predict_label(X_train_copy,X_train_copy )
    #print("加噪训练集中异常值判定阈值为：", out_clf_noise.threshold_)
    train_outliers_index_noise = []
    #print("加噪训练集样本数：", len(X_train_copy))
    for i in range(len(X_train_copy)):
        if train_pred_labels_noise[i] == 1:
            train_outliers_index_noise.append(i)
    # 训练样本中的异常值索引
    # print("加噪训练集中异常值索引：", train_outliers_index_noise)
    # print("加噪训练集中的异常值数量：", len(train_outliers_index_noise))
    # print("加噪训练集中的异常值比例：", len(train_outliers_index_noise)/len(X_train_copy))

    # subsection 从加噪测试集中检测出异常值索引
    #test_scores_noise = out_clf_noise.decision_function(X_test_copy)
    test_pred_labels_noise = out_clf_noise.predict_label(X_train_copy, X_test_copy)
    #print("加噪测试集中异常值判定阈值为：", out_clf_noise.threshold_)
    test_outliers_index_noise = []
    #print("加噪测试集样本数：", len(X_test_copy))
    for i in range(len(X_test_copy)):
        if test_pred_labels_noise[i] == 1:
            test_outliers_index_noise.append(i)
    # 训练样本中的异常值索引
    # print("加噪测试集中异常值索引：", test_outliers_index_noise)
    # print("加噪测试集中的异常值数量：", len(test_outliers_index_noise))
    # print("加噪测试集中的异常值比例：", len(test_outliers_index_noise)/len(X_test_copy))
    # SECTION softmax模型的实现

    # subsection 原始数据集上训练的softmax模型在训练集和测试集中分错的样本比例

    print("*" * 100)
    softmax_model = LogisticRegression(multi_class='ovr', solver='liblinear', random_state=42, class_weight='balanced')
    softmax_model.fit(X_train, y_train)
    train_label_pred = softmax_model.predict(X_train)
    test_label_pred = softmax_model.predict(X_test)

    # 训练样本中被softmax模型错误分类的样本
    wrong_classified_train_indices = np.where(y_train != train_label_pred)[0]
    print("训练样本中被softmax模型错误分类的样本占总训练样本的比例：",
          len(wrong_classified_train_indices) / len(y_train))

    # 测试样本中被softmax模型错误分类的样本
    wrong_classified_test_indices = np.where(y_test != test_label_pred)[0]
    print("测试样本中被softmax模型错误分类的样本占总测试样本的比例：", len(wrong_classified_test_indices) / len(y_test))

    # 整体数据集D中被softmax模型错误分类的样本
    print("完整数据集D中被softmax模型错误分类的样本占总完整数据的比例：",
          (len(wrong_classified_train_indices) + len(wrong_classified_test_indices)) / (len(y_train) + len(y_test)))

    # subsection 加噪数据集上训练的softmax模型在训练集和测试集中分错的样本比例

    print("*" * 100)
    softmax_model_noise = LogisticRegression(multi_class='ovr', solver='liblinear', random_state=42,
                                             class_weight='balanced')
    softmax_model_noise.fit(X_train_copy, y_train)
    train_label_pred_noise = softmax_model_noise.predict(X_train_copy)
    test_label_pred_noise = softmax_model_noise.predict(X_test_copy)

    # 加噪训练样本中被softmax模型错误分类的样本
    wrong_classified_train_indices_noise = np.where(y_train != train_label_pred_noise)[0]
    print("加噪训练样本中被softmax模型错误分类的样本占总加噪训练样本的比例：",
          len(wrong_classified_train_indices_noise) / len(y_train))

    # 加噪测试样本中被softmax模型错误分类的样本
    wrong_classified_test_indices_noise = np.where(y_test != test_label_pred_noise)[0]
    print("加噪测试样本中被softmax模型错误分类的样本占总测试样本的比例：",
          len(wrong_classified_test_indices_noise) / len(y_test))

    # 整体加噪数据集D中被softmax模型错误分类的样本
    print("完整数据集D中被softmax模型错误分类的样本占总完整数据的比例：",
          (len(wrong_classified_train_indices_noise) + len(wrong_classified_test_indices_noise)) / (
                      len(y_train) + len(y_test)))

    # section 修复前实验指标测定

    """Accuracy指标"""
    print("*" * 100)
    print("分类器在修复前的加噪测试集中的分类准确度：" + str(accuracy_score(y_test, test_label_pred_noise)))

    """Precision/Recall/F1指标"""

    # average='micro': 全局计算 F1 分数，适用于处理类别不平衡的情况。
    # average='macro': 类别 F1 分数的简单平均，适用于需要均衡考虑每个类别的情况。
    # average='weighted': 加权 F1 分数，适用于类别不平衡的情况，考虑了每个类别的样本量。
    # average=None: 返回每个类别的 F1 分数，适用于详细分析每个类别的表现。

    print("分类器在修复前的加噪测试集中的分类精确度：" + str(
        precision_score(y_test, test_label_pred_noise, average='weighted')))
    print("分类器在修复前的加噪测试集中的分类召回率：" + str(
        recall_score(y_test, test_label_pred_noise, average='weighted')))
    print(
        "分类器在修复前的加噪测试集中的分类F1分数：" + str(f1_score(y_test, test_label_pred_noise, average='weighted')))

    # section 识别X_copy中需要修复的元组

    # 异常检测器检测出的训练集和测试集中的异常值在原含噪数据D'中的索引
    train_outliers_noise = train_indices[train_outliers_index_noise]
    test_outliers_noise = test_indices[test_outliers_index_noise]
    outliers_noise = np.union1d(train_outliers_noise, test_outliers_noise)

    # choice 利用损失函数
    # 在加噪数据集D'上训练的softmax模型，其分类错误的样本在原含噪数据D'中的索引
    train_wrong_clf_noise = train_indices[wrong_classified_train_indices_noise]
    test_wrong_clf_noise = test_indices[wrong_classified_test_indices_noise]
    wrong_clf_noise = np.union1d(train_wrong_clf_noise, test_wrong_clf_noise)

    # outliers和分错样本的并集
    train_union = np.union1d(train_outliers_noise, train_wrong_clf_noise)
    test_union = np.union1d(test_outliers_noise, test_wrong_clf_noise)

    # 加噪数据集D'上需要修复的值
    # 需要修复的特征和标签值
    X_copy_repair_indices = outliers_noise  # 传统异常检测器仅能利用异常检测指标
    # X_copy_repair_indices = np.union1d(outliers_noise, wrong_clf_noise)

    # choice 不利用损失函数
    # X_copy_repair_indices = outliers_noise

    X_copy_repair = X_copy[X_copy_repair_indices]
    y_repair = y[X_copy_repair_indices]

    # 生成保留的行索引
    rows_to_keep = np.setdiff1d(np.arange(X_copy.shape[0]), X_copy_repair_indices)

    # 使用保留的行索引选择D'中的正常数据
    # 无需修复的特征和标签值
    X_copy_inners = X_copy[rows_to_keep]
    y_inners = y[rows_to_keep]

    # section 识别有影响力的特征
    # choice LIME(Local Interpretable Model-Agnostic Explanation)(效果好)

    # 特征数取4或6
    i = len(feature_names)
    np.random.seed(1)
    categorical_names = {}
    for feature in categorical_features:
        le = LabelEncoder()
        le.fit(data.iloc[:, feature])
        data.iloc[:, feature] = le.transform(data.iloc[:, feature])
        categorical_names[feature] = le.classes_
    explainer = LimeTabularExplainer(X_train, feature_names=feature_names, class_names=feature_names,
                                     categorical_features=categorical_features,
                                     categorical_names=categorical_names, kernel_width=3)
    # predict_proba 方法用于分类任务，predict 方法用于回归任务
    predict_fn = lambda x: softmax_model.predict_proba(x)
    exp = explainer.explain_instance(X_train[i], predict_fn, num_features=len(feature_names) // 2)
    # 获取最具影响力的特征及其权重
    top_features = exp.as_list()
    top_feature_names = [re.search(r'([a-zA-Z_]\w*)', feature[0]).group(0).strip() for feature in top_features]
    top_k_indices = [feature_names.index(name) for name in top_feature_names]
    print("LIME检验的最有影响力的属性的索引：{}".format(top_k_indices))

    # section 方案一：对X_copy中需要修复的元组进行标签修复（knn方法）
    #  需要修复的元组通过异常值检测器检测到的元组和softmax分类错误的元组共同确定（取并集）

    # subsection 尝试修复异常数据的标签

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_copy_inners, y_inners)

    # 预测异常值
    y_pred = knn.predict(X_copy_repair)

    # 替换异常值
    y[X_copy_repair_indices] = y_pred
    y_train = y[train_indices]
    y_test = y[test_indices]

    # subsection 重新在修复后的数据上训练softmax模型

    softmax_repair = LogisticRegression(multi_class='ovr', solver='liblinear', random_state=42, class_weight='balanced')
    softmax_repair.fit(X_train_copy, y_train)
    y_train_pred = softmax_repair.predict(X_train_copy)
    y_test_pred = softmax_repair.predict(X_test_copy)

    print("*" * 100)
    # 训练样本中被softmax模型错误分类的样本
    wrong_classified_train_indices = np.where(y_train != y_train_pred)[0]
    print("加噪标签修复后，训练样本中被softmax模型错误分类的样本占总训练样本的比例：",
          len(wrong_classified_train_indices) / len(y_train))

    # 测试样本中被softmax模型错误分类的样本
    wrong_classified_test_indices = np.where(y_test != y_test_pred)[0]
    print("加噪标签修复后，测试样本中被softmax模型错误分类的样本占总测试样本的比例：",
          len(wrong_classified_test_indices) / len(y_test))

    # 整体数据集D中被softmax模型错误分类的样本
    print("加噪标签修复后，完整数据集D中被softmax模型错误分类的样本占总完整数据的比例：",
          (len(wrong_classified_train_indices) + len(wrong_classified_test_indices)) / (len(y_train) + len(y_test)))

    return accuracy_score(y_test, y_test_pred),f1_score(y_test, y_test_pred, average='weighted'),precision_score(y_test, y_test_pred, average='weighted'),recall_score(y_test, y_test_pred, average='weighted')


if __name__ == '__main__':
    paths = [
        "../datasets/multi_class/drybean.csv",
        "../datasets/multi_class/obesity.csv",
        "../datasets/multi_class/balita.csv",
        "../datasets/multi_class/apple.csv",
        "../datasets/multi_class/adult.csv",
        "../datasets/multi_class/body.csv",
        #"D:/CodeWork/python/outlier/Rovas/normal_experiment/datasets/multi_class/financial/financial.csv",
        "../datasets/multi_class/online.csv",
        "../datasets/multi_class/star.csv",
        "../datasets/multi_class/Student.csv",
        "../datasets/real_outlier/Cardiotocography.csv",
        "../datasets/real_outlier/annthyroid.csv",
        "../datasets/real_outlier/optdigits.csv",
        "../datasets/real_outlier//PageBlocks.csv",
        "../datasets/real_outlier/pendigits.csv",
        "../datasets/real_outlier/satellite.csv",
        "../datasets/real_outlier/shuttle.csv",
        "../datasets/real_outlier/yeast.csv"
    ]
    #folder_path = "../datasets/real_outlier/"
    # folder_path= "../datasets/multi_class/"
    res_list = [[], [], [], [], []]
    for path in paths:
        acc,f1,precision,recall=run(path)
        res_list[0].append(str("{:.3g}".format(acc)))
        res_list[1].append(str("{:.3g}".format(f1)))
        res_list[2].append(str("{:.3g}".format(precision)))
        res_list[3].append(str("{:.3g}".format(recall)))
        res_list[4].append(path)

    resrow = list(zip(res_list[0], res_list[1], res_list[2], res_list[3],res_list[4]))
    df = pd.DataFrame(resrow, columns=['Accuracy', 'F1', 'Precision', 'Recall', 'Dataset'])

    # 保存到 CSV 文件
    df.to_csv('softmax_results_dagmm.csv', index=False)

    print("结果已保存到 results.csv 文件中")