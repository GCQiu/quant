
import mplfinance as mpf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# 读入数据
path = 'daily_data.csv'  # 数据存放路径
df = pd.read_csv(path, skiprows=0, parse_dates=['date'])
# 筛选出luid相同的行
print(df['instrument_id'])
subData = df.loc[(df['instrument_id']==104060600018)]
my_y_ticks = np.arange(0, 1, 200)
plt.xticks(my_y_ticks)
plt.figure(dpi=500,figsize=(20,10))
fig, ax = plt.subplots()
#print()
#sns.heatmap(subData.corr(method='kendall').corr(),linewidths=0.1,vmax=1.0, square=True,linecolor='white', annot=True,cmap="RdBu_r")
temp = []
open = subData['close']
for index, row in open.iteritems():
    print(row) # 输出每行的索引值
    temp.append(row)
y = []
for index, row in subData['y'].iteritems():
    print(row) # 输出每行的索引值
    y.append(row)
trends = []
for i in range(len(temp)):
    try:
        trends.append(temp[i+2]/temp[i+1]-1)
    except:
        break
print(trends,len(trends),len(temp))
ax.plot([i for i in range(len(trends))], trends, label='linear')
ax.plot([i for i in range(len(temp))], subData['y'], label='quadratic')
plt.show()
y_trends = {
    'y':y[:-2],'trends':trends
}
y_trends = pd.DataFrame(y_trends)
print(y_trends)
sns.heatmap(y_trends.corr(method='spearman').corr(),linewidths=0.1,vmax=1.0, square=True,linecolor='white', annot=True,cmap="RdBu_r")


# plt.plot([i for i in range(len(subData))],subData['y'])
plt.show()
# subData.set_index(['date'], inplace=True)
#
# # 设置k线图的风格(标题、颜色、背景)
# title = 'xingbuxing'   # 设置K线图的标题
#
# # up设置上涨颜色，down设置下跌颜色，edge设置K线边框的颜色
# my_color = mpf.make_marketcolors(up='red',down='green',edge='inherit')
#
# my_style = mpf.make_mpf_style(marketcolors=my_color)
#
# # 第一部分 绘制简单K线图
# mpf.plot(subData, type='candle', title=title, ylabel="price(usdt)", style=my_style)

