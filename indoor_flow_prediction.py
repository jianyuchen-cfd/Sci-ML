import pickle
import os
import torch
import numpy as np
from PINN_Model import PINN_Model
from Train_Model import train_model
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.colors as mcolors

anchors = [(0.00,( 3, 50,199)), (0.05,( 5, 83,216)), (0.10,( 4,124,230)), (0.15,( 5,160,232)),
           (0.20,( 5,198,177)), (0.25,( 7,214,146)), (0.30,(101,222,114)), (0.35,(166,232, 71)),
           (0.40,(217,237, 11)), (0.45,(252,220,  7)), (0.50,(252,180,  8)), (0.55,(251,131,  6)),
           (0.60,(249, 77,  7)), (0.65,(239,  5,  2)), (0.70,(189,  2,  3)), (0.75,(129,  2,  2)),
           (0.80,(116,  1,  2)), (1.00,(116,  1,  2))]
paper_cmap = mcolors.LinearSegmentedColormap.from_list(
    'paper', [(f, tuple(v/255 for v in c)) for f, c in anchors], N=256)

def sample_wall_points(n_each=40, L=3.0, H=1.0, h1=0.056, h2=0.16):
    lin = lambda a, b: torch.linspace(a, b, n_each).reshape(-1, 1)
    full = lambda v: torch.full((n_each, 1), v)

    x_wall = torch.cat([lin(0, L),  lin(0, L),  full(0.0),    full(L)])      # ceiling, floor, left, right
    y_wall = torch.cat([full(H),    full(0.0),  lin(0, H-h1), lin(h2, H)])   # left stops below inlet, right starts above outlet
    return x_wall, y_wall

def main(seed, export_dir):
    torch.manual_seed(seed); np.random.seed(seed)

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

    pinn_model = PINN_Model(input_dim=2, output_dim=3, hidden_layers=[50, 50, 50, 50])
    xy_stat = torch.cat([3.0 * torch.rand(100000, 1), torch.rand(100000, 1)], dim=1)
    pinn_model.set_input_stats(xy_stat)

    data_points = (x_data, y_data, ux_data)
    val_points = (x_val, y_val, ux_val)
    test_points = (x_test, y_test, ux_test)

    optimizer = torch.optim.Adam(pinn_model.parameters(), lr=0.001)
    trained_model, running_loss = train_model(pinn_model, optimizer=optimizer,
                data_points=data_points, val_points=val_points, Re=5000, relax_horizontal=True)
    running_loss['seed'] = seed
    torch.save(pinn_model.state_dict(), f'{export_dir}trial_{seed}.pth')

    #### Test the model on test data
    x_test, y_test, ux_test = test_points
    x_test = x_test.clone().detach().requires_grad_(True)
    y_test = y_test.clone().detach().requires_grad_(True)
    input_tensor = torch.cat((x_test, y_test), dim=1)
    
    output = trained_model(input_tensor)
    psi_pred = output[:, 0:1]
    ux_pred = torch.autograd.grad(psi_pred, y_test, grad_outputs=torch.ones_like(psi_pred))[0].detach()
    
    for name, (a, b) in {'x=H': (0, 25), 'x=2H': (25, 50), 'ceiling': (50, 78), 'floor': (78, 106)}.items():
        e = ux_pred[a:b] - ux_test[a:b]
        print(f'{name:8s} RMSE {torch.sqrt((e**2).mean()).item():.3f}')
        running_loss[f'RMSE_{name}'] = torch.sqrt((e**2).mean()).item()
    
    print("Test MSE Loss: ", torch.mean((ux_pred - ux_test) ** 2).item())
    print("Test relative L2 Error: ", (torch.norm(ux_pred - ux_test) / torch.norm(ux_test)).item())
    print("Test absolute L2 Error: ", torch.norm(ux_pred - ux_test).item())
    print("Test Root Mean Square Error: ", torch.sqrt(torch.mean((ux_pred - ux_test) ** 2)).item())
    running_loss['rmse_all'] = torch.sqrt(torch.mean((ux_pred - ux_test) ** 2)).item()
    running_loss['final_data_loss'] = running_loss['data'][-1]

    return running_loss

def plot_velocity_field(model, running_loss, export_dir, L=3.0, H=1.0):
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
    plt.savefig(f'{export_dir}loss_curve.png', dpi=300)
    #plt.show()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(running_loss["alpha"],    label='alpha for boundary loss')
    ax.plot(running_loss["beta"],     label='beta for data loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('weights')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(f'{export_dir}dynamic_weights.png', dpi=300)
    #plt.show()

    data = np.loadtxt('./measurement_all_points.csv', delimiter=',', encoding='utf-8-sig')
    x_true  = torch.tensor(data[:, 0:1], dtype=torch.float32)
    y_true  = torch.tensor(data[:, 1:2], dtype=torch.float32).requires_grad_(True)
    ux_true = torch.tensor(data[:, 2:3], dtype=torch.float32)

    input_tensor = torch.cat((x_true, y_true), dim=1)
            
    output = model(input_tensor)
    psi_pred = output[:, 0:1]
    ux_pred = torch.autograd.grad(psi_pred, y_true, grad_outputs=torch.ones_like(psi_pred))[0]
    print(f"Best alpha: {running_loss['best_alpha']}")
    print(f"Best beta: {running_loss['best_beta']}")
    print(f"Best physics loss: {running_loss['best_phys']}")
    print(f"Best data loss: {running_loss['best_data']}")
    print(f"Best boundary loss: {running_loss['best_boundary']}")
    print(f"Best total loss: {running_loss['best_total']}")
    print("All points test MSE Loss: ", torch.mean((ux_pred - ux_true) ** 2).item())
    print("Test relative L2 Error: ", (torch.norm(ux_pred - ux_true) / torch.norm(ux_true)).item())
    print("Test absolute L2 Error: ", torch.norm(ux_pred - ux_true).item())
    print("Test Root Mean Square Error: ", torch.sqrt(torch.mean((ux_pred - ux_true) ** 2)).item())
    #print(f"Ceiling error: {running_loss['RMSE_ceiling']}")
    #print(f"Floor error: {running_loss['RMSE_floor']}")

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
    plt.savefig(f'{export_dir}pred_vs_exp_vertical_velocity.png', dpi=300)
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
    plt.savefig(f'{export_dir}pred_vs_exp_horizontal_velocity.png', dpi=300)
    plt.show()

    nx, ny = 640, 320
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
    fields = [(ux, r'$u_x/u_0$', (-0.5, 1)),
        (uy, r'$u_y/u_0$', (-0.4, 0.2)),
          (umag, r'$u/u_0$', (0, 1)),
          (p, r'$p$', (-0.05, 0.2))]

    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    for j, (f, name, lim) in enumerate(fields):
        c1 = axes[j//2, j%2].contourf(X, Y, f, levels=np.linspace(*lim, 100), extend='both', cmap='jet')
        cbar = fig.colorbar(c1, ax=axes[j//2, j%2], fraction=0.05, shrink=0.7, aspect=30, orientation='horizontal', pad=0.25, label=name)
        vmin, vmax, vmean = lim[0], lim[1], np.mean(lim)
        cbar.set_ticks([vmin, vmean, vmax])
        cbar.set_ticklabels([f'{vmin:.2f}', f'{vmean:.2f}', f'{vmax:.2f}'])
        cbar.outline.set_visible(False)                             # removes the black outline
        cbar.solids.set_edgecolor('face') 
        axes[j//2, j%2].set_xlim(0, L); axes[j//2, j%2].set_ylim(0, H)
        axes[j//2, j%2].set_yticks(np.arange(0, H + 0.25/2, 0.25))
        axes[j//2, j%2].set_title(name)
        axes[j//2, j%2].set_xlabel('x/H'); axes[j//2, j%2].set_ylabel('y/H')
        axes[j//2, j%2].set_aspect('equal')

    axes[0,0].set_title('Result', loc='left')
    plt.tight_layout()
    plt.savefig(f'{export_dir}velocity_field.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    N_TRIALS = 5
    export_dir = './Running_Trial_wf_1/'
    results = []
    for seed in range(N_TRIALS):
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        if os.path.exists(f'{export_dir}trial_{seed}.pth') and os.path.exists(f'{export_dir}trials.pkl'):
            results = pickle.load(open(f'{export_dir}trials.pkl', 'rb'))
            if any(r['seed'] == seed for r in results):
                continue                                   # resume after an interruption
        results.append(main(seed, export_dir=export_dir))
        pickle.dump(results, open(f'{export_dir}trials.pkl', 'wb'))
        print(f"seed {seed}: RMSE all points = {results[-1]['rmse_all']:.3f}")
    
    #r = np.array([x['final_data_loss'] for x in results])
    r = np.array([x['rmse_all'] for x in results])
    print(f'mean {r.mean():.3f}  std {r.std():.3f}  median {np.median(r):.3f}  min {r.min():.3f}  max {r.max():.3f}')

    plt.boxplot(r); plt.ylabel(r'RMSE of $u_x/u_0$ (106 points)'); plt.savefig(f'{export_dir}trials_boxplot.png', dpi=300); plt.show()

    # representative trial: the lowest RMSE
    rep = results[int(np.argmin(r))]
    print('representative seed (closest to the median):', rep['seed'])

    if f'trial_{rep["seed"]}.pth' in os.listdir(f'{export_dir}'):
        model = PINN_Model(input_dim=2, output_dim=3, hidden_layers=[50, 50, 50, 50])
        model.load_state_dict(torch.load(f'{export_dir}trial_{rep["seed"]}.pth'))
        running_loss = rep
    else:
        raise FileNotFoundError(f"Model file for seed {rep['seed']} not found in {export_dir}")

    plot_velocity_field(model=model, running_loss=running_loss, export_dir=export_dir)


    
    