import numpy as np
import pandas as pd

# 加载数据集
# data = np.load('../datasets/Classical/1_ALOI.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/2_annthyroid.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/3_backdoor.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/4_breastw.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/5_campaign.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/6_cardio.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/7_Cardiotocography.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/8_celeba.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/9_census.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/10_cover.npz', allow_pickle=True)
data = np.load('../datasets/Classical/11_donors.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/12_fault.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/13_fraud.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/14_glass.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/15_Hepatitis.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/16_http.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/17_InternetAds.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/18_Ionosphere.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/19_landsat.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/20_letter.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/21_Lymphography.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/22_magic.gamma.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/23_mammography.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/24_mnist.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/25_musk.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/26_optdigits.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/27_PageBlocks.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/28_pendigits.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/29_Pima.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/30_satellite.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/31_satimage-2.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/32_shuttle.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/33_skin.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/34_smtp.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/35_SpamBase.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/36_speech.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/37_Stamps.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/38_thyroid.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/39_vertebral.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/40_vowels.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/41_Waveform.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/42_WBC.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/43_WDBC.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/44_Wilt.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/45_wine.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/46_WPBC.npz', allow_pickle=True)
# data = np.load('../datasets/Classical/47_yeast.npz', allow_pickle=True)

# 加载训练数据X和标签y
X, y = data['X'], data['y']

# 生成csv文件格式
# y = y.reshape(-1, 1)
# combined_numpy = np.hstack((X, y))
# # 生成列名
# feature_names = [f'feature_{i+1}' for i in range(X.shape[1])]
# column_names = feature_names + ['label']
# # 转换为 DataFrame
# df = pd.DataFrame(combined_numpy, columns=column_names)
# # 保存为 CSV 文件
# df.to_csv('E:/Rovas/Rovas_rules/baselines/multi_class_datasets/adbench_pyod_data/donors.csv', index=False)

# 使用 numpy.unique 找到唯一值
unique_values = np.unique(y)
# 计算唯一值的数量
num_unique_values = len(unique_values)
print("y属性不同的标签数量：", num_unique_values)