# SCI-ML HW#00: MLP function approximation

# 1.2 Bernstein Polynomials and a ReLU Network
#a)

import numpy as np
import matplotlib.pyplot as plt
import math

x = np.linspace(0, 1, 1000)
f1 = lambda x: np.sin(np.pi * x)
f2 = lambda x: np.abs(x - 0.5)

Bnf1 = []
Bnf2 = []

n = [4, 16, 64, 256]
for n_i in n:
    Bnf1_i = np.zeros_like(x)
    Bnf2_i = np.zeros_like(x)
    for k in range(n_i + 1):
        Bnf1_i += f1(k / n_i) * (math.comb(n_i, k) * (x ** k) * ((1 - x) ** (n_i - k)))
        Bnf2_i += f2(k / n_i) * (math.comb(n_i, k) * (x ** k) * ((1 - x) ** (n_i - k)))
    Bnf1.append(Bnf1_i)
    Bnf2.append(Bnf2_i)

print(np.abs(np.array(Bnf1) - f1(x)).max(1))
print(np.abs(np.array(Bnf2) - f2(x)).max(1))

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(x, f1(x), label='f1(x) = sin(pi*x)', color='blue')
for n_i in n:
    plt.plot(x, Bnf1[n.index(n_i)], label=f'Bernstein Polynomial n={n_i}', linestyle='--')
plt.title('Bernstein Polynomial Approximation of f1(x)=sin(pi*x)')
plt.xlabel('x')     
plt.ylabel('Value')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x, f2(x), label='f2(x) = |x - 0.5|', color='blue')
for n_i in n:
    plt.plot(x, Bnf2[n.index(n_i)], label=f'Bernstein Polynomial n={n_i}', linestyle='--')
plt.title('Bernstein Polynomial Approximation of f2(x)=|x - 0.5|')
plt.xlabel('x')
plt.ylabel('Value')
plt.legend()
plt.tight_layout()
plt.show()