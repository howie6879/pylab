# Python Tricks 03：多个界定符分割字符串

import re

line = "asdf fjdk; afed, fjek,asdf, foo"
res = re.split(r"[;,\s]\s*", line)
print(res)

# Output
# ['asdf', 'fjdk', 'afed', 'fjek', 'asdf', 'foo']
