# SCI-ML HW#00: MLP function approximation

# 1.3 Deriatives of Sigmoid and ReLU
#c)
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)
sigmoid = 1 / (1 + np.exp(-x))
deriative_sigmoid = sigmoid * (1 - sigmoid)

ReLU = np.maximum(0, x)
derivative_ReLU = (x > 0).astype(float)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(x, sigmoid, label='Sigmoid', color='blue')
plt.plot(x, deriative_sigmoid, label='Derivative of Sigmoid', color='orange')
plt.title('Sigmoid and its Derivative') 

plt.xlabel('x')
plt.ylabel('Value')     
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x, ReLU, label='ReLU', color='green')
plt.plot(x, derivative_ReLU, label='Derivative of ReLU', color='red')
plt.title('ReLU and its Derivative')
plt.xlabel('x')     
plt.ylabel('Value')
plt.legend()
plt.tight_layout()

plt.show()