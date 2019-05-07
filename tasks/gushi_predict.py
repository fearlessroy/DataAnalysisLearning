# conding:utf-8
import warnings
from datetime import datetime
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima_model import ARMA

warnings.filterwarnings('ignore')

# 中国股市走势预测，使用时间序列 ARMA
# 数据加载
df = pd.read_csv('./shanghai_1990-12-19_to_2019-2-28.csv')
# 将时间作为 df 的索引
df.Timestamp = pd.to_datetime(df.Timestamp)
df.index = df.Timestamp
# 数据探索
print(df.head())

# 按照月、季度、年来统计
df_month = df.resample('M').mean()
df_Q = df.resample('Q-DEC').mean()
df_year = df.resample('A-DEC').mean()
# 按照天，月，季度，年来显示比特币的走势
fig = plt.figure(figsize=[15, 7])
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.suptitle('中国股市', fontsize=20)
plt.subplot(221)
plt.plot(df.Price, '-', label='按天')
plt.legend()
plt.subplot(222)
plt.plot(df_month.Price, '-', label='按月')
plt.legend()
plt.subplot(223)
plt.plot(df_Q.Price, '-', label='按季度')
plt.legend()
plt.subplot(224)
plt.plot(df_year.Price, '-', label='按年')
plt.legend()
plt.show()

# 设置参数范围
ps = range(0, 3)
qs = range(0, 3)
parameters = product(ps, qs)
parameters_list = list(parameters)
# 寻找最优 ARMA 模型参数，即 best_aic 最小
results = []
best_aic = float("inf")  # 正无穷
for param in parameters_list:
    try:
        model = ARMA(df_month.Price, order=(param[0], param[1])).fit()
    except ValueError:
        print("参数错误:", param)
        continue
    aic = model.aic
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = param
    results.append([param, model.aic])
# 输出最优模型
result_table = pd.DataFrame(results)
result_table.columns = ['parameters', 'aic']
print('最优模型: ', best_model.summary())

# 股市预测
df_month2 = df_month[['Price']]
data_list = [datetime(2019, 3, 31), datetime(2019, 4, 30), datetime(2019, 5, 31), datetime(2019, 6, 30),
             datetime(2019, 7, 31),
             datetime(2019, 8, 31), datetime(2019, 9, 30), datetime(2019, 10, 31)]
future = pd.DataFrame(index=data_list, columns=df_month.columns)
df_month2 = pd.concat([df_month2, future])

df_month2['forecast'] = best_model.predict(start=0, end=348)
# print(df_month2)

# 股市预测结果显示
plt.figure(figsize=(20, 7))
df_month2.Price.plot(label='实际指数')
df_month2.forecast.plot(color='r', ls='--', label='预测指数')
plt.legend()
plt.title("中国股市（月）")
plt.xlabel('时间')
plt.ylabel('指数')
plt.show()
