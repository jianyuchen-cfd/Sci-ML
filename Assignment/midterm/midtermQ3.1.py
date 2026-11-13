import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

L = 10
EI = 5e6
k = 1e5
q0 = 1000
x = np.linspace(0, L, 10000)
dx = x[1] - x[0]
w = np.zeros_like(x)
p = lambda x: q0 * np.sin(np.pi * x / L)
w_analytical = lambda x: q0 * np.sin(np.pi * x / L) / (EI * (np.pi / L) ** 4 + k)
                                                              
t0 = time.perf_counter()
n = len(x) - 2                                   # unknowns: nodes 1..N-2 (w[0] = w[-1] = 0)
c = EI / dx ** 4
A2, A1, D, B1, B2 = [c] * n, [-4 * c] * n, [6 * c + k] * n, [-4 * c] * n, [c] * n   # five-point stencil, one row per node
D[0] -= c                                        # zero moment at the ends: ghost node w[-1] = -w[1]
D[-1] -= c
r = [float(v) for v in p(x[1:-1])]               # right-hand side

for i in range(n - 1):                           # forward elimination (banded Gauss, no pivoting needed: matrix is SPD)
    f = A1[i + 1] / D[i]
    D[i + 1] -= f * B1[i]
    B1[i + 1] -= f * B2[i]
    r[i + 1] -= f * r[i]
    if i + 2 < n:
        f = A2[i + 2] / D[i]
        A1[i + 2] -= f * B1[i]
        D[i + 2] -= f * B2[i]
        r[i + 2] -= f * r[i]

u = [0.0] * n                                    # back substitution
for i in range(n - 1, -1, -1):
    s = r[i]
    if i + 1 < n:
        s -= B1[i] * u[i + 1]
    if i + 2 < n:
        s -= B2[i] * u[i + 2]
    u[i] = s / D[i]
w[1:-1] = u
print(f"Finite difference took {time.perf_counter() - t0:.4f} s")


fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, w, label='Finite Difference', color='blue')
ax.plot(x, w_analytical(x), label='Analytical', color='red', linestyle='--')
ax.set_xlabel('x')
ax.set_ylabel('w')
ax.legend()
ax.set_title('Beam Deflection: Finite Difference vs Analytical Solution')
plt.show()
