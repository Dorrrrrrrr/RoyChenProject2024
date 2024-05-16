import numpy as np
import pickle
valid = [6, 7, 37, 38, 39, 40]    

train_data = np.load('/root/autodl-tmp/NTU-RGB-D/xview/val_data.npy')
with open('/root/autodl-tmp/NTU-RGB-D/xview/val_label.pkl', 'rb') as f:
    train_label = pickle.load(f)

print(len(train_data), type(train_data), train_data[0].shape, train_data.shape)
train_label = np.array(train_label[1])
print(train_label.shape)

indices = []
for idx, item in enumerate(train_label):
    if item+1 in valid:
        indices.append(idx)
    # break
print(len(indices))


# 使用索引列表从原始数组中选取元素  
train_label_ = train_label[indices]  
train_data_ = train_data[indices]  
  
# 将选取的元素保存为一个新的.npy文件  
np.save('/root/autodl-tmp/NTU-RGB-D/process_data/val_label.npy', train_label_)  
np.save('/root/autodl-tmp/NTU-RGB-D/process_data/val_data.npy', train_data_)  
  






'''
# type embedidng concat (x, y, acc) = 关节 embedding (25, 256)  -> max pooling 
#                                                               -> one person 

# 300 frame -> frame = token = 25 个关节 ->
# 2 * 300 * 25 关节 * [x, y, acc]

# entity_a = [0]
# entity_b = [1]        # [300, 25, 3] tensor
# [B, 300 , 25, 3] (fc) -> 300, 25, 128
# type emb  = (25, 128) expand -> [300, 25, 128] -> 300, 25, 256 -> transformer? 
# [300, 25, 256]


# (Batch, N 300, 25, 3, 2)

'''
