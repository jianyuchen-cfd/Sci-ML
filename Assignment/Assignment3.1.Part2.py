# SCI-ML HW#00: MLP function approximation

# 3.1 Learning and Composing Multi-Scale Functions
# Part 2 


import numpy as np
import matplotlib.pyplot as plt
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class MLP_ReLU(torch.nn.Module):

    def __init__(self, hidden_size=50, num_layers=1): 
        super(MLP_ReLU, self).__init__()
        self.hidden = torch.nn.Linear(1, hidden_size)
        self.activation = torch.nn.ReLU()
        self.output = torch.nn.Linear(hidden_size, 1)
        layer = [self.hidden, self.activation]
        for _ in range(num_layers - 1):
            layer += [torch.nn.Linear(hidden_size, hidden_size), self.activation]
        layer.append(self.output)
        self.model = torch.nn.Sequential(*layer)

    def forward(self, x):
        return self.model(x)

class MLP_Tanh(torch.nn.Module):

    def __init__(self, hidden_size=50, num_layers=1): 
        super(MLP_Tanh, self).__init__()
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

x_train = np.linspace(0,1,1000)
x_train_tensor = torch.tensor(x_train.reshape(-1,1), dtype = torch.float32)

y_linear = lambda x: 1+2*x
y_train_linear = y_linear(x_train)
y_train_linear_tensor = torch.tensor(y_train_linear.reshape(-1,1), dtype = torch.float32)

y_high = lambda x: 0.5*np.cos(20*np.pi*x)
y_train_high = y_high(x_train)
y_train_high_tensor = torch.tensor(y_train_high.reshape(-1,1), dtype = torch.float32)

y_final = lambda x: 1+2*x + 0.5*np.cos(20*np.pi*x)
y_train_final = y_final(x_train)
y_train_final_tensor = torch.tensor(y_train_final.reshape(-1,1), dtype = torch.float32)

model_relu = MLP_ReLU(hidden_size=50, num_layers=1)
model_tanh = MLP_Tanh(hidden_size=50, num_layers=3)
model_direct = MLP_Tanh(hidden_size=50, num_layers=3)

loss_relu = train_network(model_relu, x_train_tensor, y_train_linear_tensor, epochs=3000, learning_rate = 0.001)
loss_tanh = train_network(model_tanh, x_train_tensor, y_train_high_tensor, epochs=20000, learning_rate = 0.001)
loss_direct = train_network(model_direct, x_train_tensor, y_train_final_tensor, epochs=20000, learning_rate = 0.001)

print(f'Training Loss for linear function: {loss_relu[-1]:.6f}\n')
print(f'Training Loss for high-frequency function: {loss_tanh[-1]:.6f}\n')
print(f'Training Loss for final function: {loss_direct[-1]:.6f}\n')


x_test = np.linspace(0,1,2000)
x_test_tensor = torch.tensor(x_test.reshape(-1,1), dtype = torch.float32)

y_relu_pred = model_relu.forward(x_test_tensor).detach().numpy()
y_tanh_pred = model_tanh.forward(x_test_tensor).detach().numpy()
y_final_pred = y_relu_pred + y_tanh_pred

y_direct_pred = model_direct.forward(x_test_tensor).detach().numpy()

print(f'Test loss for linear function: {np.mean((y_relu_pred - y_linear(x_test).reshape(-1,1))**2):.6f}\n')
print(f'Test loss for high-frequency function: {np.mean((y_tanh_pred - y_high(x_test).reshape(-1,1))**2):.6f}\n')
print(f'Test loss for stichting function: {np.mean((y_final_pred - y_final(x_test).reshape(-1,1))**2):.6f}\n')
print(f'Test loss for final function: {np.mean((y_direct_pred - y_final(x_test).reshape(-1,1))**2):.6f}\n')

y_final_test = y_final(x_test)

fig, ax = plt.subplots(1,2, figsize=(10,5))
ax[0].plot(x_test, y_final_test, label='Target Function 1+2x+0.5cos(20πx)', color='red')
ax[0].plot(x_test, y_final_pred, label='Prediction with stichting 2 MLPs', linestyle='--')
ax[1].plot(x_test, y_final_test, label='Target Function 1+2x+0.5cos(20πx)', color='red')
ax[1].plot(x_test, y_direct_pred, label='Prediction with direct 1 MLP', linestyle='--')
ax[0].legend()
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[0].grid(True)
ax[1].legend() 
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
ax[1].grid(True)

fig1, ax1 = plt.subplots(1,1, figsize=(10,5))
ax1.plot(x_test, y_final_test, label='Target Function 1+2x+0.5cos(20πx)', color='red')
ax1.plot(x_test, y_final_pred, label='Prediction with stichting 2 MLPs', linestyle='--')
ax1.plot(x_test, y_direct_pred, label='Prediction with direct 1 MLP', linestyle='--')
ax1.legend()
ax1.set_xlim(0.2,0.8)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)

plt.show()