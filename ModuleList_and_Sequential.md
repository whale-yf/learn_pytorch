# nn.ModuleList 和 nn.Sequential之间的区别 
## nn.ModuleList类似于python中的list, 对储存的数据进行无序排列
例子： 

    ```python
    class net(nn.Module):
        def __init__(self):
            super(net,self).__init__()
            self.linear = nn.ModuleList([nn.Linear(10,10) for i in range(3)])
        def forward(self, x):
            for m in self.linear:
                x = m(x)
    ```


