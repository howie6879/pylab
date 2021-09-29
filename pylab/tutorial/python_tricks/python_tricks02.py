# Python Tricks 02：列表展开

import itertools

a = [[1, 2], [3, 4], [5, 6]]

print(list(itertools.chain.from_iterable(a)))
print(sum(a, []))
print([x for each_list in a for x in each_list])

# Output
# [1, 2, 3, 4, 5, 6]
