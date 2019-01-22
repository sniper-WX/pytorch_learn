from __future__ import print_function
import torch

# tensor initialize
x = torch.empty(3,5)
print(x)
x = torch.rand(5,3)
print(x)
x = torch.zeros(5,3, dtype=torch.long)
print(x)
x = torch.ones(5, 3)
print(x)
x = torch.tensor([[12,3,7],[11,3,2]])
print(x)
x = torch.randn_like(x, dtype=torch.float)
print(x)
x = x.new_ones((3, 5), dtype=torch.float)
print(x)
print(x.size())

# operations
y = torch.rand(3, 5)
## type1
print(x+y)
## type2
print(torch.add(x, y))
result = torch.zeros(3, 5)
torch.add(x, y, out=result)
print(result)
## type3
y.add_(x)
print(y)

# resize or reshape tensor
x = torch.rand(3, 5)

print(x.reshape(-1,3))
print(x.view(15))
print(x.view(1,15))

# convert from and to numpy
a = torch.rand(4, 4)
b = a.numpy()
print(a)
print(b)
a.add_(1)
print(a)
print(b)# a.numpy 得到的是一个引用
import numpy as np
a = np.ones((3,5))
b = torch.from_numpy(a)
print(a)
print(b)

# 采用GPU运算
if torch.cuda.is_available():
    device = torch.device('cuda')
    x.to(device)
    y.to(device)

    print('computing on gpu')
    import time
    start_time = time.time()
    z = x + y
    print('using gpu:%s' % z)
    print('cost time %sms' % (time.time()-start_time))
