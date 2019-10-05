---
layout: post
title: Install horovod for PyTorch to work with slurm on ABCI
---

Horovod is a great tool for multi-node, multi-gpu gradient synchronization. The official ABCI document does give a guide on running Tensorflow with multiple nodes, which can be found here [https://docs.abci.ai/en/apps/tensorflow/](https://docs.abci.ai/en/apps/tensorflow/).  However, to make PyTorch support horovod
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTI3NzA0MzM4N119
-->