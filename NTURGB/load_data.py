import numpy as np
import pickle
import os

# read file for training datasets and test datasets
train_data = np.load('dataset/NTU-RGB-D/xview/train_data.npy')
test_data = np.load('dataset/NTU-RGB-D/xview/val_data.npy')
# read file for training labels and test labels
with open ('dataset/NTU-RGB-D/xview/train_label.pkl', 'rb') as f:
    train_label = pickle.load(f)
with open ('dataset/NTU-RGB-D/xview/val_label.pkl', 'rb') as f:
    test_label = pickle.load(f)

# print the shape of train_data and the length of train_label[0] and train_label[1]
print(train_data.shape,len(train_label[0]),len(train_label[1]))
print(test_data.shape,len(test_label[0]),len(test_label[1]))
# train_data (37646, 3, 300, 25, 2), 3 is for xyz axis, 300 is for frame, 25 is for joint, 2 is for color (only xy)

# get the content of train_label[0] and train_label[1]
label_0 = train_label[0]
label_1 = train_label[1]
print(label_0[:5])
print(label_1[:5])

label_0 = set(label_0)
label_1 = set(label_1)
print(len(label_0))
# result: 37646
print(len(label_1))
# result: 60

# # read miss.txt file, get the labels that need to be deleted
# with open('NTURGB/miss.txt', 'r') as f:
#     miss_labels = [line.strip() for line in f]
# print(len(miss_labels))

# # traverse miss_labels, delete the corresponding items from the label list and data
# for miss_label in miss_labels:
#     if miss_label in train_label[0]:
#         index = train_label[0].index(miss_label)
#         del train_label[0][index]
#         train_data = np.delete(train_data, index, axis=0)

# save the processed labels and data
with open('train_label_processed.pkl', 'wb') as f:
    pickle.dump(train_label, f) 
print(train_data.shape)
# result: (37646, 3, 300, 25, 2)
np.save('train_data_processed.npy', train_data)