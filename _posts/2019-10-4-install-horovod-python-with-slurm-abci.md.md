---
layout: post
title: Install horovod for PyTorch to work with slurm on ABCI
---

Horovod is a great tool for multi-node, multi-gpu gradient synchronization. The official ABCI document does give a guide on running Tensorflow with multiple nodes, which can be found here [https://docs.abci.ai/en/apps/tensorflow/](https://docs.abci.ai/en/apps/tensorflow/).  However, to make horovod run with PyTorch and work with multiple nodes is not that straight-forward.

First, it turns out that Anaconda and Miniconda is not well supported in this case. The PyTorch plugin requires GCC version higher than 4.9. So finally, may be the only solution is to stick with the Python provided in module list and use `venv` for package control.

Load modules and create a python environment:
```bash
module load cuda/10.0/10.0.130 cudnn/7.6/7.6.4 nccl/2.4/2.4.8-1 python/3.6/3.6.5 openmpi/2.1.6
python3 -m venv $HOME/base
source $HOME/base/bin/activate
```

Install PyTorch
```bash




<!--stackedit_data:
eyJoaXN0b3J5IjpbOTAzMDk1ODMyLC0zMzg4NzY3MTddfQ==
-->