from itertools import permutations

import numpy as np

# a = np.random.lognormal(0.6, 1)
a=3
a1 = round(a, 1)  # 保留一位小数
a2 = a1 / 10
a3 = pow(10, a2 * 100)
a4 = pow(a3, 1 / 100)
Lp = 1 / a4
print(a)

print(Lp)
