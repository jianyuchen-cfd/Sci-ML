# SCI-ML HW#00: MLP function approximation

# 4.1 Regularization
# c)


import numpy as np
import matplotlib.pyplot as plt
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class MLP(torch.nn.Module):
    def __init__(self, hidden_size = 100):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.output = torch.nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x):
        x = x - torch.linspace(0, 2*torch.pi, self.hidden_size)
        x = torch.exp(-x**2/(2*0.5**2))
        x = self.output(x)
        return x
        
def train_network_L1(model, x_train, y_train, epochs = 20000, learning_rate = 0.001, ld = 1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model.forward(x_train)
        loss = loss_fn(y_pred, y_train) + ld * torch.sum(torch.abs(model.output.weight))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 1000 == 0:
            print(f'Epoch{epoch+1}/{epochs}, Loss:{loss.item():.6f}')
    return losses

def train_network_L2(model, x_train, y_train, epochs = 20000, learning_rate = 0.001, ld = 1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model.forward(x_train)
        loss = loss_fn(y_pred, y_train) + ld * torch.sum(torch.square(model.output.weight))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 1000 == 0:
            print(f'Epoch{epoch+1}/{epochs}, Loss:{loss.item():.6f}')
    return losses

x_train = np.linspace(0,2*np.pi,40)
x_train_tensor = torch.tensor(x_train.reshape(-1,1), dtype = torch.float32)

np.random.seed(0)  # For reproducibility
y = lambda x: np.sin(x) + np.random.normal(0, 0.1, x.shape)
y_train = y(x_train)
y_train_tensor = torch.tensor(y_train.reshape(-1,1), dtype = torch.float32)

x_test = np.linspace(0, 2*np.pi, 1000)
x_test_tensor = torch.tensor(x_test.reshape(-1,1), dtype = torch.float32)
model = MLP(hidden_size=100)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))

print(f'Training with L1 regularization, lambda={1e-2}')
losses_L1 = train_network_L1(model, x_train_tensor, y_train_tensor, epochs=20000, learning_rate=0.001, ld=0.01)
print(f'Training loss: {losses_L1[-1]:.6f}')
y_pred_L1 = model.forward(x_test_tensor).detach().numpy()
print(f'Test loss: {np.mean((y_pred_L1 - np.sin(x_test).reshape(-1,1))**2):.6f}')

v = model.output.weight.detach().numpy().ravel()   # shape (n,)
count = np.sum(np.abs(v) <= 1e-3)
print(f'Number of zero weights in the output layer with L1 regularization: {count} out of {len(v)}')

ax1.stem(v)
ax1.set_xlabel("output weight $w_i$ index i")
ax1.set_ylabel("output weight $w_i$")
ax1.set_title(f'L1 Regularization, lambda={1e-2}')
ax1.grid(True)

print(f'Training with L2 regularization, lambda={1e-2}')
losses_L2 = train_network_L2(model, x_train_tensor, y_train_tensor, epochs=20000, learning_rate=0.001, ld=0.01)
print(f'Training loss: {losses_L2[-1]:.6f}')
y_pred_L2 = model.forward(x_test_tensor).detach().numpy()
print(f'Test loss: {np.mean((y_pred_L2 - np.sin(x_test).reshape(-1,1))**2):.6f}')

v = model.output.weight.detach().numpy().ravel()   # shape (n,)
count = np.sum(np.abs(v) <= 1e-3)
print(f'Number of zero weights in the output layer with L2 regularization: {count} out of {len(v)}')

ax2.stem(v)
ax2.set_xlabel("output weight $w_i$ index i")
ax2.set_ylabel("output weight $w_i$")
ax2.set_title(f'L2 Regularization, lambda={1e-2}')
ax2.grid(True)

plt.show()