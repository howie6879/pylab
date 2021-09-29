# Python Tricks 04：过滤序列元素

source_list = [1, 4, -5, 10, -7, 2, 3, -1]

print([n for n in source_list if n > 0])

# 返回一个生成器
for i in (n for n in source_list if n > 0):
    print(i)


# 利用 filter 函数
def filter_func(val):
    return True if val > 0 else False


print(list(filter(filter_func, source_list)))

# Output
# [1, 4, 10, 2, 3]
# 1
# 4
# 10
# 2
# 3
# [1, 4, 10, 2, 3]
