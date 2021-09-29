# Python Tricks 06：**kwargs 的⽤法


def greet_me(**kwargs):
    for key, value in kwargs.items():
        print("{0} = {1}".format(key, value))


greet_me(name="yasoob")

# Output
# name = yasoob
