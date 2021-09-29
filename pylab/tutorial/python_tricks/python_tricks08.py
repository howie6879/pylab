# Python Tricks 08：删除字符串中不需要的字符

# Whitespace stripping
s = " hello world \n"
s.strip()
# Output 'hello world'

s.lstrip()
# Output 'hello world \n'

s.rstrip()
# Output ' hello world'

# Character stripping
t = "-----hello====="
t.lstrip("-")
# Output 'hello====='

t.strip("-=")
# Output 'hello'
