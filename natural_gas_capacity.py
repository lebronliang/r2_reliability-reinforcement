import networkx as nx  # 导入 NetworkX 工具包
import pandas as pd
import numpy as np


user = []
path = r''
df = pd.DataFrame(pd.read_excel(path))
df1 = df.copy(deep=True)
# df1.loc[0, 'Capacity(105Nm3/a)'] = df.iloc[0, 2] * 0.4
# df1.loc[1, 'Capacity(105Nm3/a)'] = df.iloc[1, 2] * 0.4#1号
# # df1.loc[2, 'Capacity(105Nm3/a)'] = df.iloc[2, 2] * 0.4#2号
# # df1.loc[3, 'Capacity(105Nm3/a)'] = df.iloc[3, 2] * 0.4
# # df1.loc[4, 'Capacity(105Nm3/a)'] = df.iloc[4, 2] * 0.4
# # df1.loc[5, 'Capacity(105Nm3/a)'] = df.iloc[5, 2] * 0.4#3号
# # df1.loc[8, 'Capacity(105Nm3/a)'] = df.iloc[8, 2] * 0.4
# # df1.loc[9, 'Capacity(105Nm3/a)'] = df.iloc[9, 2] * 0.4#4号

G1 = nx.DiGraph()  # 创建一个有向图 DiGraph
for k in range(18):
    G1.add_edge(df1.iloc[k, 0], df1.iloc[k, 1], capacity=round(df1.iloc[k, 2]),weight=round(df1.iloc[k, 4]))
mincostFlow = nx.max_flow_min_cost(G1, 1, 14, capacity='capacity', weight='weight')