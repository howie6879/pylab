# Python Tricks 10：多判断的多种写法

x, y, z = 0, 1, 0

if x == 1 or y == 1 or z == 1:
    print("passed")

if 1 in (x, y, z):
    print("passed")

if x or y or z:
    print("passed")

if any((x, y, z)):
    print("passed")
