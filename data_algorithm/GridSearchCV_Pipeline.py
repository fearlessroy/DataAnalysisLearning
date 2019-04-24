# -*- encoding=utf-8 -*-
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


'''
GridSearchCV 工具对模型参数进行调优
Pipeline 管道机制进行流水作业
'''

# 使用 RandomForest 对 IRIS 数据集进行分类
# 利用 GridSearchCV 寻找最优参数
rf = RandomForestClassifier()
parameters = {"randomforestclassifier__n_estimators": range(1, 11)}

iris = load_iris()

# 使用管道机制进行流水线作业
pipeline = Pipeline([('scaler', StandardScaler()),
                      ('randomforestclassifier', rf)
                      ])


# 使用 GridSearchCV 进行参数调优
clf = GridSearchCV(estimator=pipeline, param_grid=parameters)
# 对 iris 数据集进行分类
clf.fit(iris.data, iris.target)
print("最优分数：{:.4f}".format(clf.best_score_))
print("最优参数：{}".format(clf.best_params_))
