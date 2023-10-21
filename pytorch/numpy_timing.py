import time
import numpy as np

n = 1024 + 16

gflop = n * n * n * 2 / 1e9

a = np.random.random((n, n))
b = np.random.random((n, n))

start = time.monotonic()
for _ in range(10):
    c = a @ b
end = time.monotonic()

print(f"{gflop} GFlop")
print(f"{(end-start)/10.0:.2f} seconds per iteration")
print(f"{gflop/(end-start)*10.0:.2f} GFlop/s")
