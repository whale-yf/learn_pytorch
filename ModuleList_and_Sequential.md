# nn.ModuleList 和 nn.Sequential之间的区别 
__nn.ModuleList类似于python中的list, 对储存的数据进行无序排列__

我们先看一个nn.ModuleList的例子： 

```python
class net1(nn.Module):
    def __init__(self):
        super(net1,self).__init__()
        self.linear = nn.ModuleList([nn.Linear(10,10) for i in range(3)])
    def forward(self, x):
        for m in self.linear:
            x = m(x)
        return x
net = net1()
print(net)
print(list(net.parameters()))
# net1(
#   (linear): ModuleList(
#     (0): Linear(in_features=10, out_features=10, bias=True)
#     (1): Linear(in_features=10, out_features=10, bias=True)
#     (2): Linear(in_features=10, out_features=10, bias=True)
#   )
# )
# net1(
#   (linear): ModuleList(
#     (0): Linear(in_features=10, out_features=10, bias=True)
#     (1): Linear(in_features=10, out_features=10, bias=True)
#     (2): Linear(in_features=10, out_features=10, bias=True)
#   )
# )
# [Parameter containing:
```
我们可以看出ModuleList与python结构是类似的，都是用索引来指向参数

ok,我们在来看一下如果用list来储存这些结构效果：
```python
class net1(nn.Module):
    def __init__(self):
        super(net1,self).__init__()
        self.linear = [nn.Linear(10,10) for i in range(3)]
    def forward(self, x):
        for m in self.linear:
            x = m(x)
        return x
net = net1()
print(net)
print(list(net.parameters()))
# net1()
# []
```
这里我们可以看出用list作为容器，返回值中是没有创建的方法在里面的，这也说明了nn.ModuleList在创建一系列方法后，还携带了实现这些方法的接口，这样理解对吗？

接下来我们看一下nn.Sequential：
```python
class net2(nn.Module):
    def __init__(self):
        super(net2,self).__init__()
        self.linear = nn.Sequential(
            nn.Conv2d(4,2,kernel_size=2,stride=1,padding=0),
            nn.BatchNorm2d(5),
            nn.LeakyReLU()
        )
    def forward(self, x):
        
        return self.linear(x)
net = net2()
print(net)
print(list(net.parameters()))
# net1(
#   (linear): Sequential(
#     (0): Conv2d(4, 2, kernel_size=(2, 2), stride=(1, 1))
#     (1): BatchNorm2d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (2): LeakyReLU(negative_slope=0.01)
#   )
# )
# [Parameter containing:
```
这里我们可以简单的对比以下nn.ModuleList和nn.Sequential，nn.ModuleList在调用方法的时候，用了for循环，而nn.Sequential在调用的时候没有使用for循环，直接给出了其中所有的方法。由此可以看出nn.ModuleList在调用方法的时候是无序调用的，简单点理解为可以按照索引来对nn.ModuleList中的方法进行调用，nn.Sequential在调用的时候是不可以按照索引进行调用的，而是按照顺序进行调用。 

总结：
nn.ModuleList可以理解为一个存放各种方法的List,而且能够调用这些方法，用法上相比于nn.Sequential更加灵活，但是在操作上就需要更多的操作。
nn.Sequential可以理解为对nn.ModuleList中的方法，按照输入的顺序进行封装，可以在一些固定方法中使用。



