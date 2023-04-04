import einops
import torch
import torch.nn.functional as F
x = torch.arange(256*18*18).reshape((256,18,18))


tile = torch.ones((256,2),dtype=int)


i = tile[:,0]
j = tile[:,1]

print(i.size())
print(j.size())
x = x[torch.arange(256),i,j]
print(x.size())
print(x[0:20])

# print(x[0])
# print(x[1])
