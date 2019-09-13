# Testing Numba


## Installation on Ubuntu 18.04

1. Install CUDA on your machine (if possible): https://docs.nvidia.com/cuda/cuda-installation-guide-linux
1. Install Numba: `conda install numba`

Following this [tutorial](https://devblogs.nvidia.com/numba-python-cuda-acceleration/)

## Test Numba

```python
import numpy as np
from numba import vectorize

@vectorize(['float32(float32, float32)'], target='cuda')
def Add(a, b):
  return a + b

# Initialize arrays
N = 100000
A = np.ones(N, dtype=np.float32)
B = np.ones(A.shape, dtype=A.dtype)
C = np.empty_like(A, dtype=A.dtype)

# Add arrays on GPU
C = Add(A, B)
```