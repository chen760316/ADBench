import nibabel as nib
from sklearn.metrics import accuracy_score

from adbench.baseline.PyOD import PYOD
import numpy as np
from sklearn.model_selection import train_test_split

data = np.load('../datasets/Classical/6_cardio.npz', allow_pickle=True)
X, y = data['X'], data['y']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = PYOD(seed=42, model_name='XGBOD')  # initialization
model.fit(X_train=X_train, y_train=y_train)  # fit
score = model.predict_score(X_test)  # predict
threshold = 0.8

# 获取大于阈值的元素的下标
indices = np.where(score > threshold)[0]
y_pred = np.copy(y_test)
# 首先将整个 y_pred 置为 0
y_pred[:] = 0
y_pred[indices] = 1
print("测试集中异常值的分类准确度：" + str(accuracy_score(y_test, y_pred)))