# SCI-ML HW#00: MLP function approximation

# 1.2 Bernstein Polynomials and a ReLU Network
#b) use a single neural network

import numpy as np
import matplotlib.pyplot as plt
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class ReLU_Net(torch.nn.Module):

    def __init__(self, hidden_size=10): 
        super(ReLU_Net, self).__init__()
        self.hidden = torch.nn.Linear(1, hidden_size)
        #torch.nn.init.normal_(self.hidden.weight, mean=0.0, std=1.0)
        with torch.no_grad():
            kinks = torch.rand(hidden_size)     # kink positions in [0, 1]
            self.hidden.bias.copy_(-self.hidden.weight.squeeze(1) * kinks)
        self.output = torch.nn.Linear(hidden_size, 1)
        #torch.nn.init.normal_(self.output.weight, mean=0.0, std=0.1)
        #torch.nn.init.zeros_(self.output.bias)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.hidden(x)
        x = self.activation(x)
        x = self.output(x)
        return x

def train_network(model, x_train, y_train, epochs=3000, learning_rate=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model.forward(x_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())    
    return losses

x_test = np.linspace(0, 1, 1000)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).reshape(-1, 1)
x_train_np = np.linspace(0, 1, 200)
y_train1 = np.sin(np.pi * x_train_np)
y_train2 = np.abs(x_train_np - 0.5)

f1 = lambda x: np.sin(np.pi * x)
f2 = lambda x: np.abs(x - 0.5)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 4))
ax1.set_title('Function Approximation: sin(pi*x)')
ax2.set_title('Function Approximation: |x - 0.5|')

ax1.plot(x_test, f1(x_test), label='f1(x) = sin(pi*x)', color='blue')
ax2.plot(x_test, f2(x_test), label='f2(x) = |x - 0.5|', color='blue')

for n_i in [4, 16, 64, 256]:
    x_train = torch.tensor(x_train_np, dtype=torch.float32).reshape(-1, 1)
    y1_train = torch.tensor(f1(x_train_np), dtype=torch.float32).reshape(-1, 1)
    y2_train = torch.tensor(f2(x_train_np), dtype=torch.float32).reshape(-1, 1)

    model1 = ReLU_Net(hidden_size=n_i)
    model2 = ReLU_Net(hidden_size=n_i)
    losses1 = train_network(model1, x_train, y1_train, epochs=3000, learning_rate=0.002)
    losses2 = train_network(model2, x_train, y2_train, epochs=3000, learning_rate=0.002)

    y1_pred = model1.forward(x_test_tensor).detach().numpy()
    print(f'n={n_i}: sin loss = {np.abs(y1_pred - f1(x_test).reshape(-1, 1)).max()}')
    ax1.plot(x_test_tensor.numpy(), y1_pred, label=f'ReLU Network n={n_i}', linestyle='--')
    y2_pred = model2.forward(x_test_tensor).detach().numpy()
    print(f'n={n_i}: abs loss = {np.abs(y2_pred - f2(x_test).reshape(-1, 1)).max()}')
    ax2.plot(x_test_tensor.numpy(), y2_pred, label=f'ReLU Network n={n_i}', linestyle='--')

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Function Approximation: sin(pi*x)')
ax1.legend()
ax1.grid(True)

ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Function Approximation: |x - 0.5|')
ax2.legend()
ax2.grid(True)

plt.show()