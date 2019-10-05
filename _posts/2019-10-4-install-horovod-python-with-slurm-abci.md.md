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
pip install numpy
pip install torch==0.4.1
```

Install horovod with NCCL support and a higher version of g++
```bash
CC=/apps/gcc/7.3.0/bin/gcc CXX=/apps/gcc/7.3.0/bin/g++ HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_NCCL_HOME=/apps/nccl/2.4.8-1/cuda10.0/lib pip install horovod
```

To test whether the horovod works or not, let's make a simple script named `hvd_test.py` .

```python
import horovod.torch as hvd
import torch
hvd.init()
print(
	"size=", hvd.size(),
	"global_rank=", hvd.rank(),
	"local_rank=", hvd.local_rank(),
	"device=", torch.cuda.get_device_name(hvd.local_rank())
)
```

The launch the job, make a script like this:

```bash
#!/bin/bash

source /etc/profile.d/modules.sh
module load cuda/10.0/10.0.130 cudnn/7.6/7.6.4 nccl/2.4/2.4.8-1 python/3.6/3.6.5 openmpi/2.1.6
source $HOME/base/bin/activate

NUM_NODES=${NHOSTS}
NUM_GPUS_PER_NODE=4
NUM_GPUS_PER_SOCKET=$(expr ${NUM_GPUS_PER_NODE} / 2)
NUM_PROCS=$(expr ${NUM_NODES} \* ${NUM_GPUS_PER_NODE})

LD_LIBRARY_PATH=/apps/gcc/7.3.0/lib64:$LD_LIBRARY_PATH

mpirun -np $NUM_PROCS --map-by ppr:${NUM_GPUS_PER_SOCKET}:socket \
--mca mpi_warn_on_fork 0 -x PATH -x LD_LIBRARY_PATH \
python hvd_test.py > hvd_test.stdout
```



<!--stackedit_data:
eyJoaXN0b3J5IjpbMTA5NDk0MzQxMSwtMTQwOTg4NjUsLTMzOD
g3NjcxN119
-->