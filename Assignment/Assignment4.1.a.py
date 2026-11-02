# SCI-ML HW#00: MLP function approximation

# 4.1 Regularization
# a)


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
        
def train_network(model, x_train, y_train, epochs = 20000, learning_rate = 0.001):
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
        if epoch % 1000 == 0:
            print(f'Epoch{epoch+1}/{epochs}, Loss:{loss.item():.6f}')
    return losses

x_train = np.linspace(0,2*np.pi,40)
x_train_tensor = torch.tensor(x_train.reshape(-1,1), dtype = torch.float32)

np.random.seed(0)
y = lambda x: np.sin(x) + np.random.normal(0, 0.1, x.shape)
y_train = y(x_train)
y_train_tensor = torch.tensor(y_train.reshape(-1,1), dtype = torch.float32)

x_test = np.linspace(0, 2*np.pi, 1000)
x_test_tensor = torch.tensor(x_test.reshape(-1,1), dtype = torch.float32)
model = MLP(hidden_size=100)
losses = train_network(model, x_train_tensor, y_train_tensor, epochs=20000, learning_rate=0.001)
print(f'Training loss: {losses[-1]:.6f}')
y_pred = model.forward(x_test_tensor).detach().numpy()
print(f'Test loss: {np.mean((y_pred - np.sin(x_test).reshape(-1,1))**2):.6f}')

fig, ax = plt.subplots(1,1, figsize=(5,5))
ax.scatter(x_train, y_train, color='blue', label='Training Data', marker='o', s=10)
ax.plot(x_test, np.sin(x_test), color='green', label='Target Function')
ax.plot(x_test, y_pred, color='red', label='Predicted Data', linestyle='--')
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Fit with no penalty')
plt.show()