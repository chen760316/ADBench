#!/usr/bin/env python
# coding: utf-8

# **Step-by-step Guidence on How to Install and Use ADBench**

# # Install ADBench

# In[ ]:


# In[ ]:


# download datasets in ADBench from the remote github repo
import nibabel as nib
from adbench.myutils import Utils
utils = Utils()
# we recommend jihulab for China mainland user and github otherwise
utils.download_datasets(repo='jihulab')


# # Run ADBench 

# ## Run ADBench experiments

# In[ ]:


from adbench.run import RunPipeline

'''
Params:
suffix: file name suffix;

parallel: running either 'unsupervise', 'semi-supervise', or 'supervise' (AD) algorithms,
corresponding to the Angle I: Availability of Ground Truth Labels (Supervision);

realistic_synthetic_mode: testing on 'local', 'global', 'dependency', and 'cluster' anomalies, 
corresponding to the Angle II: Types of Anomalies;

noise type: evaluating algorithms on 'duplicated_anomalies', 'irrelevant_features' and 'label_contamination',
corresponding to the Angle III: Model Robustness with Noisy and Corrupted Data.
'''


# In[ ]:


# return the results including [params, model_name, metrics, time_fit, time_inference]
# besides, results will be automatically saved in the dataframe and ouputted as csv file in adbench/result folder

pipeline = RunPipeline(suffix='ADBench', parallel='semi-supervise', realistic_synthetic_mode=None, noise_type=None)
results = pipeline.run()


# In[ ]:


pipeline = RunPipeline(suffix='ADBench', parallel='unsupervise', realistic_synthetic_mode='cluster', noise_type=None)
results = pipeline.run()


# In[ ]:


pipeline = RunPipeline(suffix='ADBench', parallel='supervise', realistic_synthetic_mode=None, noise_type='irrelevant_features')
results = pipeline.run()


# ## Run your customized algorithm on ADBench datasets

# In[ ]:


# customized model on ADBench's datasets
from adbench.run import RunPipeline
from adbench.baseline.Customized.run import Customized

# notice that you should specify the corresponding category of your customized AD algorithm
# for example, here we use Logistic Regression as customized clf, which belongs to the supervised algorithm
# for your own algorithm, you can realize the same usage as other baselines by modifying the fit.py, model.py, and run.py files in the adbench/baseline/Customized
pipeline = RunPipeline(suffix='ADBench', parallel='supervise', realistic_synthetic_mode=None, noise_type=None)
results = pipeline.run(clf=Customized)


# ## Run your customized algorithm on customized dataset

# In[ ]:


# customized model on customized dataset
import numpy as np
dataset = {}
dataset['X'] = np.random.randn(1000, 20)
dataset['y'] = np.random.choice([0, 1], 1000)
results = pipeline.run(dataset=dataset, clf=Customized)


# # Import AD algorithms from ADBench

# In[ ]:


import numpy as np
X_train = np.random.randn(1000, 20)
y_train = np.random.choice([0, 1], 1000)
X_test = np.random.randn(100, 20)

# Directly import AD algorithms from the existing toolkits like PyOD
from adbench.baseline.PyOD import PYOD
model = PYOD(seed=42, model_name='XGBOD')  # initialization
model.fit(X_train, y_train)  # fit
score = model.predict_score(X_test)  # predict

# Import deep learning AD algorithms from our ADBench
from adbench.baseline.PReNet.run import PReNet
model = PReNet(seed=42)
model.fit(X_train, y_train)  # fit
score = model.predict_score(X_test)  # predict

