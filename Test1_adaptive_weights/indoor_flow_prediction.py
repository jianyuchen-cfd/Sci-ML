import pickle
import os
import torch
import numpy as np
from PINN_Model import PINN_Model
from Train_Model import train_model
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def sample_wall_points(n_each=40, L=3.0, H=1.0, h1=0.056, h2=0.16):
    lin = lambda a, b: torch.linspace(a, b, n_each).reshape(-1, 1)
    full = lambda v: torch.full((n_each, 1), v)

    x_wall = torch.cat([lin(0, L),  lin(0, L),  full(0.0),    full(L)])      # ceiling, floor, left, right
    y_wall = torch.cat([full(H),    full(0.0),  lin(0, H-h1), lin(h2, H)])   # left stops below inlet, right starts above outlet
    return x_wall, y_wall

def main():
    MODEL_PATH = 'pinn_model.pth'
    LOSS_PATH  = 'running_loss.pkl'
    pinn_model = PINN_Model(input_dim=2, output_dim=2, hidden_layers=[50, 50, 50, 50])

    if os.path.exists(MODEL_PATH):
        pinn_model.load_state_dict(torch.load(MODEL_PATH))
        with open(LOSS_PATH, 'rb') as f:
            running_loss = pickle.load(f)
        print('Loaded trained model, skipping training.')

        return pinn_model
    else:
        data = np.loadtxt('./measurement.csv', delimiter=',', encoding='utf-8-sig')
        x_data  = torch.tensor(data[:, 0:1], dtype=torch.float32)
        y_data  = torch.tensor(data[:, 1:2], dtype=torch.float32)
        ux_data = torch.tensor(data[:, 2:3], dtype=torch.float32)
        val_data = np.loadtxt('./validation_data.csv', delimiter=',', encoding='utf-8-sig')
        x_val  = torch.tensor(val_data[:, 0:1], dtype=torch.float32)
        y_val  = torch.tensor(val_data[:, 1:2], dtype=torch.float32)
        ux_val = torch.tensor(val_data[:, 2:3], dtype=torch.float32)
        test_data = np.loadtxt('./measurement_all_points.csv', delimiter=',', encoding='utf-8-sig')
        x_test  = torch.tensor(test_data[:, 0:1], dtype=torch.float32)
        y_test  = torch.tensor(test_data[:, 1:2], dtype=torch.float32)
        ux_test = torch.tensor(test_data[:, 2:3], dtype=torch.float32)

        pinn_model.set_input_stats(torch.cat([x_data, y_data], dim=1))

        data_points = (x_data, y_data, ux_data)
        val_points = (x_val, y_val, ux_val)
        test_points = (x_test, y_test, ux_test)

        wall_points = sample_wall_points()

        optimizer = torch.optim.Adam(pinn_model.parameters(), lr=0.001)
        trained_model, running_loss = train_model(pinn_model, optimizer=optimizer,
                    data_points=data_points, val_points=val_points, Re=5000)


        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(running_loss["total"],    label='total loss')
        ax.plot(running_loss["phys"],     label='physics loss')
        ax.plot(running_loss["data"],     label='data loss')
        ax.plot(running_loss["boundary"], label='boundary loss')
        ax.set_yscale('log')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig('loss_curve.png', dpi=300)
        plt.show()

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(running_loss["alpha"],    label='alpha for boundary loss')
        ax.plot(running_loss["beta"],     label='beta for data loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('weights')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig('dynamic_weights.png', dpi=300)
        plt.show()

        torch.save(pinn_model.state_dict(), MODEL_PATH)
        with open(LOSS_PATH, 'wb') as f:
            pickle.dump(running_loss, f)

        #### Test the model on test data
        x_test, y_test, ux_test = test_points
        x_test = x_test.clone().detach().requires_grad_(True)
        y_test = y_test.clone().detach().requires_grad_(True)
        input_tensor = torch.cat((x_test, y_test), dim=1)
        
        output = trained_model(input_tensor)
        psi_pred = output[:, 0:1]
        ux_pred = torch.autograd.grad(psi_pred, y_test, grad_outputs=torch.ones_like(psi_pred))[0].detach()
        print("Test MSE Loss: ", torch.mean((ux_pred - ux_test) ** 2).item())
        print("Test relative L2 Error: ", (torch.norm(ux_pred - ux_test) / torch.norm(ux_test)).item())
        print("Test absolute L2 Error: ", torch.norm(ux_pred - ux_test).item())
        print("Test Root Mean Square Error: ", torch.sqrt(torch.mean((ux_pred - ux_test) ** 2)).item())

        return trained_model

def plot_velocity_field(model, L=3.0, H=1.0):
    data = np.loadtxt('./measurement_all_points.csv', delimiter=',', encoding='utf-8-sig')
    x_true  = torch.tensor(data[:, 0:1], dtype=torch.float32)
    y_true  = torch.tensor(data[:, 1:2], dtype=torch.float32).requires_grad_(True)
    ux_true = torch.tensor(data[:, 2:3], dtype=torch.float32)

    input_tensor = torch.cat((x_true, y_true), dim=1)
            
    output = model(input_tensor)
    psi_pred = output[:, 0:1]
    ux_pred = torch.autograd.grad(psi_pred, y_true, grad_outputs=torch.ones_like(psi_pred))[0]
    print("All points test MSE Loss: ", torch.mean((ux_pred - ux_true) ** 2).item())
    print("Test relative L2 Error: ", (torch.norm(ux_pred - ux_true) / torch.norm(ux_true)).item())
    print("Test absolute L2 Error: ", torch.norm(ux_pred - ux_true).item())
    print("Test Root Mean Square Error: ", torch.sqrt(torch.mean((ux_pred - ux_true) ** 2)).item())

    idx = [25, 50, 78, 106, 0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 8))
    for i in [0,1]:
        axes[i].plot(ux_true[idx[i-1]:idx[i]], y_true[idx[i-1]:idx[i]].detach().numpy(), marker='o', color='k',label='Experiment')
        axes[i].plot(ux_pred[idx[i-1]:idx[i]].detach().numpy(), y_true[idx[i-1]:idx[i]].detach().numpy(), color='k', linestyle='--', label='PINN Prediction')
        axes[i].set_xlabel(r'$u_x/u_0$')
        axes[i].set_ylabel(r'y=x/H')
        axes[i].set_ylim(0, 1)
        axes[i].set_xlim(-0.5, 1)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        if i == 0:
            axes[i].set_title('x=H')
        else:
            axes[i].set_title('x=2H')   
    plt.savefig('pred_vs_exp_vertical_velocity.png', dpi=300)
    plt.show()

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    for i, j in enumerate([2,3]):
        axes[i].plot(x_true[idx[j-1]:idx[j]].detach().numpy(), ux_true[idx[j-1]:idx[j]], marker='o', color='k',label='Experiment')
        axes[i].plot(x_true[idx[j-1]:idx[j]].detach().numpy(), ux_pred[idx[j-1]:idx[j]].detach().numpy(), color='k', linestyle='--', label='PINN Prediction')
        axes[i].set_xlabel(r'x/H')
        axes[i].set_ylabel(r'$u_x/u_0$')
        axes[i].set_xlim(0, 3)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        if i == 0:
            axes[i].set_title('y=H-0.5h')
            axes[i].set_ylim(-0.5, 1.5)
        else:
            axes[i].set_title('y=0.5h') 
            axes[i].set_ylim(-0.5, 0.5)
    plt.savefig('pred_vs_exp_horizontal_velocity.png', dpi=300)
    plt.show()

    nx, ny = 600, 400
    xg = np.linspace(0, L, nx)
    yg = np.linspace(0, H, ny)
    X, Y = np.meshgrid(xg, yg)                                   # shape (ny, nx)

    x_t = torch.tensor(X.reshape(-1, 1), dtype=torch.float32, requires_grad=True)
    y_t = torch.tensor(Y.reshape(-1, 1), dtype=torch.float32, requires_grad=True)

    out = model(torch.cat([x_t, y_t], dim=1))
    psi, p = out[:, 0:1], out[:, 1:2]
    ux =  torch.autograd.grad(psi, y_t, torch.ones_like(psi), retain_graph=True)[0]
    uy = -torch.autograd.grad(psi, x_t, torch.ones_like(psi))[0]

    ux = ux.detach().numpy().reshape(ny, nx)
    uy = uy.detach().numpy().reshape(ny, nx)
    p  = p.detach().numpy().reshape(ny, nx)
    umag = np.sqrt(ux**2 + uy**2)

    p      = p - p.mean()
    fields = [(ux, r'$u_x/u_0$', (0, 1)),
        (uy, r'$u_y/u_0$', (-0.4, 0.2)),
          (umag, r'$u/u_0$', (0, 1)),
          (p, r'$p$', (-0.05, 0.2))]

    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    for j, (f, name, lim) in enumerate(fields):
        c1 = axes[j//2, j%2].contourf(X, Y, f, levels=np.linspace(*lim, 50), cmap='jet', extend='both')
        cbar = fig.colorbar(c1, ax=axes[j//2, j%2], fraction=0.05, shrink=0.7, aspect=30, orientation='horizontal', pad=0.25, label=name)
        vmin, vmax, vmean = lim[0], lim[1], np.mean(lim)
        cbar.set_ticks([vmin, vmean, vmax])
        cbar.set_ticklabels([f'{vmin:.2f}', f'{vmean:.2f}', f'{vmax:.2f}'])
        axes[j//2, j%2].set_xlim(0, L); axes[j//2, j%2].set_ylim(0, H)
        axes[j//2, j%2].set_title(name)
        axes[j//2, j%2].set_xlabel('x/H'); axes[j//2, j%2].set_ylabel('y/H')
        axes[j//2, j%2].set_aspect('equal')

    axes[0,0].set_title('Result', loc='left')
    plt.tight_layout()
    plt.savefig('velocity_field.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    trained_model = main()
    plot_velocity_field(trained_model)


    
    