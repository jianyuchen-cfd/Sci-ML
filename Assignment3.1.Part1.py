# SCI-ML HW#00: MLP function approximation

# 3.1 Learning and Composing Multi-Scale Functions
# Part 1 


import numpy as np
import matplotlib.pyplot as plt
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class MLP(torch.nn.Module):

    def __init__(self, hidden_size=50, num_layers=1): 
        super(MLP, self).__init__()
        self.hidden = torch.nn.Linear(1, hidden_size)
        self.activation = torch.nn.Tanh()
        self.output = torch.nn.Linear(hidden_size, 1)
        layer = [self.hidden, self.activation]
        for _ in range(num_layers - 1):
            layer += [torch.nn.Linear(hidden_size, hidden_size), self.activation]
        layer.append(self.output)
        self.model = torch.nn.Sequential(*layer)

    def forward(self, x):
        return self.model(x)

def train_network(model, x_train, y_train, epochs=3000, learning_rate = 0.01):
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

x_train = np.linspace(0,1,400)
x_train_tensor = torch.tensor(x_train.reshape(-1,1), dtype = torch.float32)

y_high = lambda x: 0.5*np.cos(20*np.pi*x)
y_train = y_high(x_train)
y_train_tensor = torch.tensor(y_train.reshape(-1,1), dtype = torch.float32)

x_train_test = np.linspace(0,1,2000)
x_train_test_tensor = torch.tensor(x_train_test.reshape(-1,1), dtype = torch.float32)


fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2,2, figsize=(10,5))
ax = [ax1, ax2, ax3, ax4]
epochs = [10000, 15000, 30000, 30000]
for i,j in [(1,1), (2,2), (3,3), (4,4)]:
    print(f'Training MLP with {j} Layers\n')
    model = MLP(hidden_size=50, num_layers=j)
    loss = train_network(model, x_train_tensor, y_train_tensor, epochs=epochs[j-1], learning_rate = 0.001)
    print(f'Training Loss for {j} Layers: {loss[-1]:.6f}\n')
    y_pred = model.forward(x_train_test_tensor).detach().numpy()
    print(f'Test loss for {j} Layers: {np.mean((y_pred - y_high(x_train_test).reshape(-1,1))**2):.6f}\n')
    
    ax[i-1].plot(x_train_test, y_pred, label=f'Prediction with {j} Layer(s)', linestyle='--')
    ax[i-1].plot(x_train_test, y_high(x_train_test), label='Target Function 0.5cos(20πx)', color='red')
    ax[i-1].legend()
    ax[i-1].set_xlabel('x')
    ax[i-1].set_ylabel('y')
    ax[i-1].grid(True)

plt.show()


