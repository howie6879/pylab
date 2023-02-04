# Python Tricks 01：合并两个字典

# Python3.5+
x = {"a": 1, "b": 2}
y = {"b": 3, "c": 4}

z1 = {**x, **y}

# Python 2, (or 3.4 or lower)
def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


z2 = merge_two_dicts(x, y)

# Output
# {'a': 1, 'b': 3, 'c': 4}
# 欢迎订阅公众号：老胡的储物柜
