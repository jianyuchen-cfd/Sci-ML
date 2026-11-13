import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def FDM_solver():
    x = np.linspace(0, 1, 1000)
    t = np.linspace(0, 2, 2001)
    u = np.zeros((len(x), len(t)))
    c = 1.0

    u0 = np.sin(np.pi * x) + 0.5 * np.sin(3 * np.pi * x)
    cfl = c * (t[1]-t[0]) / (x[1]-x[0])
    u[:,0] = u0
    u[1:-1,1]= u[1:-1,0] + 0.5 * cfl ** 2 * (u[2:,0] - 2*u[1:-1,0] + u[:-2,0])

    for i in range(1, len(t)-1):

        u[1:-1,i+1] = ((t[1]-t[0])) ** 2 * c**2 * (u[2:,i] - 2*u[1:-1,i] + u[:-2,i]) / ((x[1]-x[0]))**2 + 2*u[1:-1,i] - u[1:-1,i-1]

    fig, axes = plt.subplots(2,3, figsize=(20,10))
    for i, j in enumerate(np.arange(0,2.5,0.5)):
        axes[1][2].contourf(t, x, u)
        t_index = t.tolist().index(j)
        axes[i//3][i%3].plot(x,u[:,t_index])

    plt.show()
    return u

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_layers[0]))
        layers.append(nn.Tanh())
        
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_layers[-1], output_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

    def derivative(self, xt):
        x = xt[:,0:1].detach().clone().requires_grad_(True)
        t = xt[:,1:2].detach().clone().requires_grad_(True)
        q = torch.cat((x, t), dim=1)
        u = self.forward(q)
        du_dx = torch.autograd.grad(outputs=u, inputs=x,
                                    grad_outputs=torch.ones_like(u), create_graph=True)[0]
        du_dt = torch.autograd.grad(outputs=u, inputs=t,
                                    grad_outputs=torch.ones_like(u),
                                    create_graph=True)[0]
        du2_dx2 = torch.autograd.grad(outputs=du_dx, inputs=x,
                                      grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]
        du2_dt2 = torch.autograd.grad(outputs=du_dt, inputs=t,
                                      grad_outputs=torch.ones_like(du_dt), create_graph=True)[0]
        return du_dx, du_dt, du2_dx2, du2_dt2

def train_model(model, x_train, y_train, epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = {'total_loss': [], 'physical_loss': [], 'initial_condition_x_loss': [], 'initial_condition_t_loss': [], 'boundary_condition_loss': []}
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train).reshape(nx,nt)
        du_dx, du_dt, du2_dx2, du2_dt2 = model.derivative(x_train)
        du_dt = du_dt.reshape(nx,nt)
        physical_loss = criterion(du2_dt2, c**2 * du2_dx2)  # Example physical loss, modify as needed
        initial_condition_x_loss_ = criterion(outputs[:,0], y_train)  # Example initial condition loss, modify as needed
        initial_condition_t_loss_ = criterion(du_dt[:,0], torch.zeros_like(du_dt[:,0]))  # Example initial condition loss, modify as needed
        boundary_condition_loss =criterion(outputs[0,:], torch.zeros_like(outputs[0,:])) + criterion(outputs[-1,:], torch.zeros_like(outputs[-1,:]))  # Example boundary condition loss, modify as needed
        alpha, lambda_1, lambda_2, beta = 1.0, 1.0, 1.0, 1.0  # Example weights for the loss components, modify as needed
        loss = alpha * physical_loss + lambda_1 * initial_condition_x_loss_ + lambda_2 * initial_condition_t_loss_ + beta * boundary_condition_loss
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
            print(f'Physical Loss: {physical_loss.item():.4f}, Initial Condition x Loss: {initial_condition_x_loss_.item():.4f}, Initial Condition t Loss: {initial_condition_t_loss_.item():.4f}, Boundary Condition Loss: {boundary_condition_loss.item():.4f}')
        losses['total_loss'].append(loss.item())
        losses['physical_loss'].append(physical_loss.item())
        losses['initial_condition_x_loss'].append(initial_condition_x_loss_.item())
        losses['initial_condition_t_loss'].append(initial_condition_t_loss_.item())
        losses['boundary_condition_loss'].append(boundary_condition_loss.item())
    return model,losses
c = 1.0  # Example wave speed, modify as needed
nx , nt = 100, 200  # Number of spatial and temporal points
x = torch.linspace(0, 1, nx)
t = torch.linspace(0, 2, nt)
X, T = torch.meshgrid(x, t, indexing='ij')
x_train = torch.stack([X.reshape(-1), T.reshape(-1)], dim=1) 
print(f'x_train shape: {x_train.shape}')  # Debugging line to check the shape of x_train
model = MLP(input_dim=2, output_dim=1, hidden_layers=[50, 50, 50])
y = lambda  x: np.sin(np.pi * x) + 0.5 * np.sin(3 * np.pi * x)  # Example function for initial condition, modify as needed
y_train = y(x)
trained_model, losses = train_model(model, x_train, y_train, epochs=1000, learning_rate=0.001)

x_test = torch.linspace(0, 1, 1000)
t_test = torch.linspace(0, 2, 2001)
X, T = torch.meshgrid(x_test, t_test, indexing='ij')
x_test = torch.stack([X.reshape(-1), T.reshape(-1)], dim=1) 
with torch.no_grad():
    u_pred = trained_model(x_test).detach().numpy().reshape(1000, 2001)
u = FDM_solver()
x = np.linspace(0, 1, 1000)
t = np.linspace(0, 2, 2001)

fig, ax = plt.subplots(1,1, figsize=(10,6))
ax.plot(losses['total_loss'], label='Total Loss')
ax.plot(losses['physical_loss'], label='Physical Loss')
ax.plot(losses['initial_condition_x_loss'], label='Initial Condition x Loss')
ax.plot(losses['initial_condition_t_loss'], label='Initial Condition t Loss')   
ax.plot(losses['boundary_condition_loss'], label='Boundary Condition Loss')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.set_title('Training Losses')

fig, axes = plt.subplots(2,3, figsize=(20,10))
for i, j in enumerate(np.arange(0,2.5,0.5)):
    axes[1][2].contourf(t, x, u_pred)
    t_index = t.tolist().index(j)
    axes[i//3][i%3].plot(x,u_pred[:,t_index])

plt.show()

fig, axes = plt.subplots(1,3, figsize=(20,10))
t_index_2 = t.tolist().index(2.0)
error = np.abs(u_pred - u)
axes[0].contourf(t, x, error)
axes[1].contourf(t, x, u_pred)
axes[2].contourf(t, x, u, label='FDM')
axes[0].set_title('Error at t=2.0')
axes[1].set_title('Predicted Solution at t=2.0')
axes[2].set_title('FDM Solution at t=2.0')
error = np.abs(u_pred[:,t_index_2] - u[:,t_index_2])
print(f'root mean squared error at t=2.0: {np.sqrt(np.mean(error**2))}')
