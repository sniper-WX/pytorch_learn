import torch
'''
Tensor:
    设置requires_grad,为True时，会跟踪所有在此tensor中的operation
    调用.backward()可计算所有的梯度（导数），并将本tensor的梯度存入.grad属性中
    调用.detach()将此tensor在跟踪历史中去除，在后续的跟踪不会再记录其operation
    将代码放在with torch.no_grad():中，可以停止所有代码块中tensor对operation的跟踪
    .grad_fn属性是指向Function的一个引用，该Function记录了此tensor是如何计算出来的。
Funtion：
    Funtion和Tensor共同构成一张有向无环图，共同表征了一个完整的计算过程。tensor代表节点，.grad_fn代表边。
'''

#设置requires_grad为false，计算过程不会被记录，tensor的grad_fn属性为None
x = torch.ones(2,2,requires_grad=False)
print(x)
y = x**2 - 2*x
print(y)

#设置requires_grad为True，计算过程会被记录
x = torch.ones(2,2,requires_grad=True)
print(x)
y = x**2 - 2*x
print(y)
x.requires_grad_(False)#初始化之后设置requires_grad属性，
z = 2*y  # y虽然基于x计算出来，但y的计算过程仍然会被跟踪
z1 = 2*x  # x参与作为自变量的计算不会再被跟踪，
print(z)
print(z1)

# 梯度（导数）计算,调用backward（），只有自己定义的变量才会有grad属性，在计算图中，只有叶子节点的tensor有可访问的梯度值。
# 下面的例子中，定义了x，并基于x计算y，y.backward传入的张量代表的是y的梯度取值dy，梯度可反向传到最终计算出x的梯度

x = torch.tensor([[0.7230, 0.1206],
        [0.9032, 0.8995]], requires_grad=True)
c = 2*x-2 # 计算导数在x处的取值
print("c: %s"%c)
y = x**2 - x*2
y.backward(torch.ones(2, 2)) # y的梯度取1
print(x.grad)
y.backward(torch.ones(2, 2)+1) # y的梯度取2，y的梯度与x的梯度具有线性正相关的关系
print(x.grad)

