import seaborn as sns  
import numpy as np  
import matplotlib.pyplot as plt  
  
# 创建一个示例矩阵  
data = np.random.rand(10, 12)  # 10x12的随机矩阵，值在0到1之间  
  
data = np.array([[302,  11,  15,   3,   4,   3],
        [  8, 268,   2,  44,   5,  18],
        [  3,   0, 270,   4,  12,   3],
        [  2,  23,   3, 256,  12,  14],
        [  1,   9,  26,   7, 249,  28],
        [  0,   5,   0,   2,  34, 246]])

# 绘制热力图  
# 使用 Seaborn 绘制混淆矩阵的热力图
sns.heatmap(data, annot=True, fmt='d', cmap='Blues')

# 设置标题和轴标签
plt.title('Confusion Matrix')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')


# plt.xlabel('横坐标标签')  
# plt.ylabel('纵坐标标签')  
# 如果你的数据是DataFrame，并且已经有索引和列名，则这些标签会自动设置  
  


 
  
# 显示图形  
plt.show()
plt.savefig("热力图.jpg")