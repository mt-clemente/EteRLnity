from math import comb

n = 16

n_size = 0

for k in range(0,16):
    n_size += k ** 2


for k in range(7,14):
    print(comb(k**2,2))
    n_size += comb(k**2,2)


print(n_size)