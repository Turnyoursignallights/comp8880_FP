import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# 读取数据
# 假设 features.csv 文件中每一行代表一个用户的特征数据
features_data1 = pd.read_csv("data/large_twitch_features.csv")
edges_data1 = pd.read_csv("data/large_twitch_edges.csv")

edges_data1 
edges_df = edges_df_sampled.sample(frac=sample_percentage, random_state=1)

# 假设这些变量已经根据实际数据定义好了
target_user_id = 24516
top5_ids = [144643, 128864, 96473, 61862, 129896]



# 初始化网络图G
G = nx.Graph()

# 基于edges.csv添加边
for _, row in edges_data.iterrows():
    G.add_edge(row['numeric_id_1'], row['numeric_id_2'])

# 绘制网络图
plt.figure(figsize=(8,6))
nx.draw(G, with_labels=True, node_size=700, node_color='lightblue')
plt.title("Network Graph")
plt.show()