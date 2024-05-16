
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple
from roy_transformer import *
from roy_module import fc_block, build_normalization




class ARModel(nn.Module):
    """
    Overview:
        Implementation of the Transformer model.

    .. note::
        For more details, refer to "Attention is All You Need": http://arxiv.org/abs/1706.03762.

    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        head_dim: int = 128,
        hidden_dim: int = 1024,
        output_dim: int = 256,
        head_num: int = 2,
        mlp_num: int = 2,
        layer_num: int = 3,
        dropout_ratio: float = 0.,
        activation: nn.Module = nn.ReLU(),
    ):
        """
        Overview:
            Initialize the Transformer with the provided dimensions, dropout layer, activation function,
            and layer numbers.
        Arguments:
            - input_dim (:obj:`int`): The dimension of the input.
            - head_dim (:obj:`int`): The dimension of each head in the multi-head attention mechanism.
            - hidden_dim (:obj:`int`): The dimension of the hidden layer in the MLP (Multi-Layer Perceptron).
            - output_dim (:obj:`int`): The dimension of the output.
            - head_num (:obj:`int`): The number of heads in the multi-head attention mechanism.
            - mlp_num (:obj:`int`): The number of layers in the MLP.
            - layer_num (:obj:`int`): The number of Transformer layers.
            - dropout_ratio (:obj:`float`): The dropout ratio for the dropout layer.
            - activation (:obj:`nn.Module`): The activation function used in the MLP.
        """
        super(ARModel, self).__init__()
        self.type_embedding = nn.Embedding(25, input_dim // 2)                  # 低维离散表征 -> 高维稠密向量 [128]
        self.state_embedding = fc_block(state_dim, input_dim // 2, activation=activation) # [3] -> fc -> [128]  
        self.embedding = fc_block(input_dim, output_dim, activation=activation)
        self.decoder = fc_block(output_dim, 60, activation=activation)      

        self.act = activation
        layers = []
        dims = [output_dim] + [output_dim] * layer_num
        self.dropout = nn.Dropout(dropout_ratio)
        for i in range(layer_num):
            layers.append(
                TransformerLayer(dims[i], head_dim, hidden_dim, dims[i + 1], head_num, mlp_num, self.dropout, self.act)
            )
        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Overview:
            Perform the forward pass through the Transformer.
        Arguments:

            - x (:obj:`torch.Tensor`): The input tensor, with shape `(B, N, V, C, M)`, 
                where `B` is batch size, \
                      `N` is the number of entries 图片帧数 300.
                      `V` is 关节数量 25.
                      `C` is 关节特征数量 3 (x, y, acc).
                      `M` is 人数 2.
                (Batch, 300, 25, 3, 2)      
            - mask (:obj:`Optional[torch.Tensor]`, optional): The mask tensor (bool), used to mask out invalid \
                entries in attention. It has shape `(B, N)`, where `B` is batch size and `N` is number of \
                entries. Defaults to None.
        Returns:
            - x (:obj:`torch.Tensor`): The output tensor from the Transformer.
        """
        # (B, 300, 25, 3, 2) -> (B, 300, 256)
        B, N, V, C, M = x.shape
        enta, entb = torch.split(x, split_size_or_sections=1, dim=-1)
        state_emb_a = self.state_embedding(enta.view(-1, 300, 25, 3))
        state_emb_b = self.state_embedding(entb.view(-1, 300, 25, 3))                         # (B, 300, 25, 128)

        indices = torch.arange(25).unsqueeze(0).unsqueeze(0).repeat(B, 300, 1).to('cuda')           
        type_emb = self.type_embedding(indices)                                               # (B, 300, 25, 128)

        enta_emb = torch.concat([state_emb_a, type_emb], dim = -1)                            # (B, 300, 25, 256)          
        enta_emb = enta_emb.mean(dim = -2)                                                    # (B, 300, 256)
        entb_emb = torch.concat([state_emb_b, type_emb], dim = -1)
        entb_emb = entb_emb.mean(dim = -2)                                                    # (B, 300, 256)

        x = enta_emb + entb_emb                                                               # (B, 300, 256)

        if mask is not None:
            mask = mask.unsqueeze(dim=1).repeat(1, mask.shape[1], 1).unsqueeze(dim=1)
        x = self.embedding(x)
        x = self.dropout(x)
        x, mask = self.main((x, mask))                                                         # (b, 300, 256)                                            
        
        x = x.mean(dim = -2)                                                                   # (B, 256)
        logits = self.decoder(x)                                                               # (B, 60)
        return logits 

