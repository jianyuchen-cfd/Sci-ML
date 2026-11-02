# SCI-ML HW#00: MLP function approximation

# 2.1 Approximating a discoutinuous function


import numpy as np
import matplotlib.pyplot as plt
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class SingleLayer_Net(torch.nn.Module):

    def __init__(self, hidden_size=50): 
        super(SingleLayer_Net, self).__init__()
        self.hidden_1 = torch.nn.Linear(1, hidden_size)
#        self.hidden_2 = torch.nn.Linear(hidden_size, hidden_size)
#        self.hidden_3 = torch.nn.Linear(hidden_size, hidden_size)
        self.output = torch.nn.Linear(hidden_size, 1)
        self.activation = torch.nn.Tanh()   
#        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.hidden_1(x)
        x = self.activation(x)
#        x = self.hidden_2(x)
#        x = self.activation(x)
#        x = self.hidden_3(x)
#        x = self.activation(x)
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
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}')
    return losses

x_train = np.linspace(-1, 1, 200)
y_train = np.where(x_train < 0.5, 0, 1)

x_test = np.linspace(-1, 1, 1000)
y_test = np.where(x_test < 0.5, 0, 1)

x_train_tensor = torch.tensor(x_train.reshape(-1, 1), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
x_test_tensor = torch.tensor(x_test.reshape(-1, 1), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

model = SingleLayer_Net(hidden_size=500)
train_network(model, x_train_tensor, y_train_tensor, epochs=3000, learning_rate=0.005)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x_test, y_test, label='Target Function', color='blue')
y_pred = model.forward(x_test_tensor).detach().numpy()
ax.plot(x_test_tensor.numpy(), y_pred, label='MLP Approximation with L1Loss', color='orange')
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Function Approximation')
plt.show()
