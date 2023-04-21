#!/bin/sh
pip install wandb einops protobuf==3.20.*
wandb login cdd836c352ffd933807c80225c7b616d7ba369d7
ssh-keygen
cat /home/ubuntu/.ssh/id_rsa.pub
