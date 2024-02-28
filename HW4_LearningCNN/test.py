import torch

# 创建两个张量
tensor1 = torch.tensor([1, 2, 3, 4, 5])
tensor2 = torch.tensor([1, 4, 5, 6, 7])

# 计算相等元素的数量

print(tensor1 == tensor2)
equal_elements_count = torch.sum(tensor1 == tensor2).item()

print("相等元素的数量为:", equal_elements_count)
