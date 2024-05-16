import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, Dataset
from ar_model import ARModel

hashmap = {
    6-1: 0,
    7-1: 1,
    37-1: 2,
    38-1: 3,
    39-1: 4,
    40-1: 5
}
# 定义一个简单的数据集
class JointDataset(Dataset):
    def __init__(self, datas, labels):
        self.datas = datas                # (Batch, 300, 25, 3, 2)
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


# 创建数据集

# TestData
val_data = np.load('/root/autodl-tmp/NTU-RGB-D/process_data/val_data.npy')
print(val_data.shape)
val_label = np.load('/root/autodl-tmp/NTU-RGB-D/process_data/val_label.npy')

print(val_data.shape, val_label.shape)
val_dataset = JointDataset(val_data, val_label)
val_dataloader = DataLoader(val_dataset, batch_size = 512)


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
model_weights = torch.load('model_weights.pth')  
model.load_state_dict(model_weights)  
if torch.cuda.is_available():  
    model.cuda()
    print("load on gpu")

def test():
    matrix = torch.zeros([6, 6], dtype=torch.long).to('cuda')  # 真正例  

    for i, batch in enumerate(val_dataloader):
            inputs = batch['inputs'].to('cuda')
            labels = batch['labels'].to('cuda')
            logits = model(inputs)             # (Batch, 60)

            _, predicted = torch.max(logits, 1)
            for i, pred in enumerate(predicted):
                matrix[hashmap[pred.item()]][hashmap[labels[i].item()]] += 1
                    
            correct = (predicted == labels).float()  
            
            # 计算准确率（正确的预测数 / 总样本数）  
            acc = correct.sum() / len(labels)

            print(f"ACC: {acc}")
            
    print(matrix)        
test()


def calculate_metrics(labels, predicted):  
    num_classes = max(labels.max().item(), predicted.max().item()) + 1  # 假设类别索引从0开始  
    tp = torch.zeros(num_classes, dtype=torch.long)  # 真正例  
    fp = torch.zeros(num_classes, dtype=torch.long)  # 假正例  
    fn = torch.zeros(num_classes, dtype=torch.long)  # 假反例  
  
    for i in range(num_classes):  
        tp[i] = (labels == i) & (predicted == i).sum().item()  # 真正例：真实类别和预测类别都为i的样本数  
        fp[i] = (labels != i) & (predicted == i).sum().item()  # 假正例：真实类别不为i但预测为i的样本数  
        fn[i] = (labels == i) & (predicted != i).sum().item()  # 假反例：真实类别为i但预测不为i的样本数  
  
    return tp, fp, fn  

def test_mul(num_classes):  
    tp = torch.zeros(num_classes, dtype=torch.long).to('cuda')  # 真正例  
    fp = torch.zeros(num_classes, dtype=torch.long).to('cuda')  # 假正例  
    fn = torch.zeros(num_classes, dtype=torch.long).to('cuda')  # 假反例  
    tn_like = torch.zeros(num_classes, dtype=torch.long).to('cuda')  # 假反例  
  
    for i, batch in enumerate(val_dataloader):  
        inputs = batch['inputs'].to('cuda')  
        labels = batch['labels'].to('cuda')  
          
        # 假设模型的输出logits是一个形状为(batch_size, num_classes)的tensor  
        logits = model(inputs)  
        _, predicted = torch.max(logits, 1)  # 获取预测类别的索引  
          
        # 计算每个类别的TP, FP, FN  
        for c in range(num_classes):  
            if c+1 in [6, 7, 37, 38, 39, 40]:    
                tp[c] += ((predicted == c) & (labels == c)).sum().item()  # 真正例  
                fp[c] += ((predicted == c) & (labels != c)).sum().item()  # 假正例  
                fn[c] += ((predicted != c) & (labels == c)).sum().item()  # 假反例  
            
        # （可选）计算并打印当前批次的准确率  
        correct = (predicted == labels).float().sum().item()  
        total = labels.size(0)  
        acc = correct / total if total > 0 else 0  
        print(f"Batch {i+1} ACC: {acc}", total)  
  
    # （可选）打印每个类别的TP, FP, FN  
    total = 0
    for c in range(num_classes):  
        if c+1 in [6, 7, 37, 38, 39, 40]:
            total += tp[c] + fp[c] + fn[c] 

            for other_c in range(num_classes):  
                if other_c != c:  
                    tn_like[c] += ((predicted != c) & (labels == other_c)).sum().item()  
            print(f"Class {c}: TP={tp[c]}, FP={fp[c]}, FN={fn[c]}, TN = {tn_like[c]}")    


    print(total)
    # （可选）计算并打印总体准确率（所有类别都被正确分类的样本比例）  
    # 注意：这通常不是多分类任务中评估模型性能的主要指标  
    total_correct = tp.sum().item()  
    total_samples = len(val_dataloader.dataset)  # 假设dataloader知道数据集的总大小  
    overall_acc = total_correct / total_samples if total_samples > 0 else 0  
    print(f"Overall ACC: {overall_acc}")  
test_mul(60)