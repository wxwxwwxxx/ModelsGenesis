import torch
from torch.nn import functional as F

a = torch.randn([2,3,4,5])  # torch.Size([2, 3, 4, 5])
padding = (
    1,2,   # 前面填充1个单位，后面填充两个单位，输入的最后一个维度则增加1+2个单位，成为8
    2,3,
    3,4
)
print(a.shape)
b = F.pad(a, padding)
print(b.shape)  # torch.Size([2, 10, 9, 8])
