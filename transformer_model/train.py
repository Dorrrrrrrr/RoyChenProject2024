import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, Dataset
from ar_model import ARModel

class JointDataset(Dataset):
    def __init__(self, datas, labels):
        self.datas = datas                # (Batch 3w, 300, 25, 3(x, y, acc), 2)
        self.labels = labels              # (Batch, )

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        inputs = self.datas[idx]
        labels = self.labels[idx]
        return {
            'inputs': torch.tensor(inputs, dtype=torch.float),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


# 训练数据集
train_data = np.load('/root/autodl-tmp/NTU-RGB-D/process_data/train_data.npy')
train_label = np.load('/root/autodl-tmp/NTU-RGB-D/process_data/train_label.npy')
print(train_data.shape, train_label.shape)
dataset = JointDataset(train_data, train_label)
dataloader = DataLoader(dataset, batch_size = 512, shuffle = True )

# 测试数据集
val_data = np.load('/root/autodl-tmp/NTU-RGB-D/process_data/val_data.npy')
val_label = np.load('/root/autodl-tmp/NTU-RGB-D/process_data/val_label.npy')
print(val_data.shape, val_label.shape)
val_dataset = JointDataset(val_data, val_label)
val_dataloader = DataLoader(val_dataset, batch_size = 512, shuffle = False)


# 初始化Transformer模型
model = ARModel(
            state_dim = 3, 
            input_dim = 256,  
            head_dim = 128,
            hidden_dim = 256,
            output_dim = 256,
            head_num = 2,
            mlp_num = 2,
            layer_num = 1,
        )

#model_weights = torch.load('model_weights.pth')  
#model.load_state_dict(model_weights)  

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
decayRate = 0.96

# from torch.optim.lr_scheduler import ExponentialLR
# my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=decayRate)

optimizer = torch.optim.Adam(model.parameters(), lr= 1e-5)

# 训练模型
model.train()
if torch.cuda.is_available():  
    model.cuda()
    print("load on gpu")


def test():
   #  with torch.no_grad:
        for i, batch in enumerate(val_dataloader):
                inputs = batch['inputs'].to('cuda')
                labels = batch['labels'].to('cuda')
                logits = model(inputs)             # (Batch, 60)

                _, predicted = torch.max(logits, 1)
                correct = (predicted == labels).float()  
                
                # 计算准确率（正确的预测数 / 总样本数）  
                acc = correct.sum() / len(labels)

                print(f"ACC: {acc}")
            
import matplotlib.pyplot as plt  
losses = []
cnt = 0
for epoch in range(100000000000000000000):  
    for batch in dataloader:
        inputs = batch['inputs'].to('cuda')
        labels = batch['labels'].to('cuda')
        # print("---", inputs.shape, labels.shape)
        optimizer.zero_grad()
        logits = model(inputs)             # (Batch, )
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}", cnt)
        
        if cnt % 100 == 0:
            torch.save(model.state_dict(), 'model_weights.pth')
            test()
            losses.append(loss.item())  
            # 在训练结束后绘制损失曲线图  
            plt.plot(losses)  
            plt.xlabel('Epoch')  # 如果你想按epoch显示，可以修改为'Epoch'并使用epoch // len(dataloader)作为x轴值  
            plt.ylabel('Loss')  
            plt.title('Loss Curve')  
      
            plt.savefig('loss_curve.jpg')
        cnt += 1

        if loss.item() < 0.01:
            break
    