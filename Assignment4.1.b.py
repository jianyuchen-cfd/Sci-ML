# SCI-ML HW#00: MLP function approximation

# 4.1 Regularization
# b)


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

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))

ax1.scatter(x_train, y_train, color='blue', label='Training Data', marker='o', s=10)
ax1.plot(x_test, np.sin(x_test), color='green', label='Target Function')

ax2.scatter(x_train, y_train, color='blue', label='Training Data', marker='o', s=10)
ax2.plot(x_test, np.sin(x_test), color='green', label='Target Function')

for ld in [1e-4, 1e-3, 1e-2, 1e-1]:
    print(f'Training with L1 regularization, lambda={ld}')
    losses_L1 = train_network_L1(model, x_train_tensor, y_train_tensor, epochs=20000, learning_rate=0.001, ld=ld)
    print(f'Training loss: {losses_L1[-1]:.6f}')
    y_pred_L1 = model.forward(x_test_tensor).detach().numpy()
    print(f'Test loss: {np.mean((y_pred_L1 - np.sin(x_test).reshape(-1,1))**2):.6f}')

    ax1.plot(x_test, y_pred_L1, label=f'Predicted Data with lambda={ld}, RMSE: {np.sqrt(np.mean((y_pred_L1 - np.sin(x_test).reshape(-1,1))**2)):.6f}', linestyle='--')

    print(f'Training with L2 regularization, lambda={ld}')
    losses_L2 = train_network_L2(model, x_train_tensor, y_train_tensor, epochs=20000, learning_rate=0.001, ld=ld)
    print(f'Training loss: {losses_L2[-1]:.6f}')
    y_pred_L2 = model.forward(x_test_tensor).detach().numpy()
    print(f'Test loss: {np.mean((y_pred_L2 - np.sin(x_test).reshape(-1,1))**2):.6f}')

    ax2.plot(x_test, y_pred_L2, label=f'Predicted Data with lambda={ld}, RMSE: {np.sqrt(np.mean((y_pred_L2 - np.sin(x_test).reshape(-1,1))**2)):.6f}', linestyle='--')

ax1.legend(loc='lower left')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title(f'L1 Regularization, lambda={ld}')
ax1.grid(True)
ax2.legend(loc='lower left')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title(f'L2 Regularization, lambda={ld}')
ax2.grid(True)
plt.show()