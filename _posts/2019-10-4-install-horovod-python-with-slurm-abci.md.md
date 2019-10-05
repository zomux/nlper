---
layout: post
title: Install horovod for PyTorch to work with slurm on ABCI
---

Horovod is a great tool for multi-node, multi-gpu gradient synchronization. The official ABCI document does give a guide on running Tensorflow with multiple nodes, which can be found here [https://docs.abci.ai/en/apps/tensorflow/](https://docs.abci.ai/en/apps/tensorflow/).  However, to make horovod run with PyTorch and work with multiple nodes is not that straight-forward.

First, it turns out that Anaconda and Miniconda is not well supported in this case. The PyTorch plugin requires GCC version higher than 4.9. So finally, may be the only solution is to stick with the Python provided in module list and use `venv` for package control.


<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE0OTY3Mzc5LC0zMzg4NzY3MTddfQ==
-->