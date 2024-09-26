import nibabel as nib
from sklearn.metrics import accuracy_score

from adbench.baseline.PReNet.run import PReNet
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = np.load('../datasets/Classical/6_cardio.npz', allow_pickle=True)
X, y = data['X'], data['y']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = PReNet(seed=42)

# 创建给定标签比例为0.1的半监督数据集，其余训练标签未知，用？代替

# 计算要随机抽取的标签数量
n_samples_to_select = int(len(y_train) * 0.1)

# 随机选择索引
random_indices = np.random.choice(len(y_train), n_samples_to_select, replace=False)

# 创建一个新的数组，默认值为 '?'
y_l = np.full(y_train.shape, '?', dtype=object)

# 将抽取的部分标签放入新数组中
y_l[random_indices] = y_train[random_indices]

model.fit(X_train=X_train, y_train=y_l)  # fit
score = model.predict_score(X_test)  # predict

# 创建直方图
plt.hist(score, bins=30, edgecolor='black')  # bins 设置直方图的条形数量

# 添加标题和标签
plt.title('Histogram of Data Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')

# 显示图形
plt.show()

threshold = 10

# 获取大于阈值的元素的下标
indices = np.where(score > threshold)[0]
y_pred = np.copy(y_test)
# 首先将整个 y_pred 置为 0
y_pred[:] = 0
y_pred[indices] = 1
print("测试集中异常值的分类准确度：" + str(accuracy_score(y_test, y_pred)))