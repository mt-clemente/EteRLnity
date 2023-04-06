Traceback (most recent call last):
  File "/home/wsl/Polymtl/H23/INF6201/Projet/Network/train.py", line 382, in <module>
    train_model()
  File "/home/wsl/Polymtl/H23/INF6201/Projet/Network/train.py", line 210, in train_model
    optimize(
  File "/home/wsl/Polymtl/H23/INF6201/Projet/Network/train.py", line 269, in optimize
    state_values = policy_net(mask_batch.squeeze(1)).squeeze(1).gather(1,action_batch)
RuntimeError: index -2 is out of bounds for dimension 1 with size 1024
Traceback (most recent call last):
  File "/home/wsl/Polymtl/H23/INF6201/Projet/Network/train.py", line 382, in <module>
    train_model()
  File "/home/wsl/Polymtl/H23/INF6201/Projet/Network/train.py", line 210, in train_model
    optimize(
  File "/home/wsl/Polymtl/H23/INF6201/Projet/Network/train.py", line 269, in optimize
    state_values = policy_net(mask_batch.squeeze(1)).squeeze(1).gather(1,action_batch)
RuntimeError: index -2 is out of bounds for dimension 1 with size 1024
make: *** [Makefile:2: full] Error 1