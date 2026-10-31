import os
import torch
import numpy as np
from Calculate_Loss import data_loss, physic_loss, boundary_loss 
from PINN_Model import PINN_Model
from Train_Model import train_model, sample_collocation
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def sample_wall_points(n_each=10, L=3.0, H=1.0, h1=0.056, h2=0.16):
    lin = lambda a, b: torch.linspace(a, b, n_each).reshape(-1, 1)
    full = lambda v: torch.full((n_each, 1), v)

    x_wall = torch.cat([lin(0, L),  lin(0, L),  full(0.0),    full(L)])      # ceiling, floor, left, right
    y_wall = torch.cat([full(H),    full(0.0),  lin(0, H-h1), lin(h2, H)])   # left stops below inlet, right starts above outlet
    return x_wall, y_wall

def main():
    data = np.loadtxt('./measurement.csv', delimiter=',', encoding='utf-8-sig')
    x_data  = torch.tensor(data[:, 0:1], dtype=torch.float32)
    y_data  = torch.tensor(data[:, 1:2], dtype=torch.float32)
    ux_data = torch.tensor(data[:, 2:3], dtype=torch.float32)

    randperm = torch.randperm(x_data.size(0))
    idx_train = randperm[:int(0.8 * len(randperm))]
    idx_val   = randperm[int(0.8 * len(randperm)):int(0.9 * len(randperm))]
    idx_test  = randperm[int(0.9 * len(randperm)):]

    data_points = (x_data[idx_train], y_data[idx_train], ux_data[idx_train])
    val_points = (x_data[idx_val], y_data[idx_val], ux_data[idx_val])
    test_points = (x_data[idx_test], y_data[idx_test], ux_data[idx_test])

    wall_points = sample_wall_points()

    pinn_model = PINN_Model(input_dim=2, output_dim=2, hidden_layers=[50, 50, 50, 50])
    optimizer = torch.optim.Adam(pinn_model.parameters(), lr=0.001)
    trained_model, running_loss = train_model(pinn_model, optimizer=optimizer,
                boundary_points=wall_points, data_points=data_points, val_points=val_points, Re=5000)


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
    plt.show()

if __name__ == "__main__":
    main()

    
    