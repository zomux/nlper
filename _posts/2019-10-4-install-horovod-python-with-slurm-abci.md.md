---
layout: post
title: Install horovod for PyTorch to work with slurm on ABCI
---

Horovod is a great tool for multi-node, multi-gpu gradient synchronization. The official ABCI document does give a guide on running Tensorflow with multiple nodes, which can be found here [https://docs.abci.ai/en/apps/tensorflow/](https://docs.abci.ai/en/apps/tensorflow/).  However, to make horovod run with PyTorch and work with multiple nodes is not that straight-forward.

First, it turns out that Anaconda and Miniconda is not well supported in this case. The PyTorch plugin
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTMzODg3NjcxN119
-->