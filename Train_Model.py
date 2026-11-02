import copy
import torch
from Calculate_Loss import data_loss, physic_loss, boundary_loss


def sample_wall_points(n=40, L=3.0, H=1.0, h1=0.056, h2=0.16):
    lengths = torch.tensor([L, L, H - h1, H - h2])
    seg = torch.multinomial(lengths / lengths.sum(), n, replacement=True)   # which wall each point is on
    t = torch.rand(n)                                                       # random position along that wall

    x = torch.zeros(n); y = torch.zeros(n)
    m = seg == 0; x[m] = L * t[m];  y[m] = H                       # ceiling
    m = seg == 1; x[m] = L * t[m];  y[m] = 0.0                     # floor
    m = seg == 2; x[m] = 0.0;       y[m] = (H - h1) * t[m]         # left wall, below the inlet
    m = seg == 3; x[m] = L;         y[m] = h2 + (H - h2) * t[m]    # right wall, above the outlet
    return x.reshape(-1, 1), y.reshape(-1, 1)

def sample_collocation(n=640, n_wall=0, L=3.0, H=1.0, strip=0.1):
    # uniform points over the whole room
    x = L * torch.rand(n, 1)
    y = H * torch.rand(n, 1)

    # n_wall extra points in a thin strip along each of the four walls
    xs, ys = [x], [y]

    xs.append(L * torch.rand(n_wall, 1));             ys.append(H - strip * torch.rand(n_wall, 1))   # ceiling
    xs.append(L * torch.rand(n_wall, 1));             ys.append(strip * torch.rand(n_wall, 1))       # floor
    xs.append(strip * torch.rand(n_wall, 1));         ys.append(H * torch.rand(n_wall, 1))           # left wall
    xs.append(L - strip * torch.rand(n_wall, 1));     ys.append(H * torch.rand(n_wall, 1))           # right wall

    return torch.cat(xs), torch.cat(ys)

def train_model(model, optimizer,  data_points, val_points, Re,
             n_collocation_points=640, L=3.0, H=1.0, num_epochs=3000):
    """
    Train the PINN model using physics-informed loss and boundary condition loss.

    Parameters:
    - model: The neural network model (PINN).
    - optimizer: The optimizer for training.
    - val_points: The validation points for evaluating the model.
    - collocation_points: The collocation points for the physics-informed loss.
    - data_points: The data points for the data loss.
    - Re: Reynolds number
    - num_epochs: Number of training epochs

    Returns:
    - model: The trained model.
    """
    x_d, y_d, ux_d = data_points
    x_v, y_v, ux_v = val_points

    best_val, best_state = float('inf'), None
    running_loss = {"phys": [], "data": [], "boundary": [], "total": [], 'alpha': [], 'beta': []}

    def grad_stats(loss, params):
        g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        g = torch.cat([gi.flatten() for gi in g if gi is not None])
        return g.abs().max(), g.abs().mean()

    alpha, beta, a = 1.0, 150.0, 0.1
    params = [p for p in model.parameters()]

    for epoch in range(num_epochs):
        x_c, y_c = sample_collocation(n=n_collocation_points, L=L, H=H)  
        x_b, y_b = sample_wall_points()
        # Calculate physics-informed loss
        phys_loss = physic_loss(model, x_c, y_c, Re)
        # Calculate data loss (assuming you have data)
        data_loss_value = data_loss(model, x_d, y_d, ux_d)

        # Calculate boundary condition loss (assuming you have boundary data)
        boundary_loss_value = boundary_loss(model, x_b, y_b)
        """ TEST1: Update the weights for the different loss components """
        if epoch % 10 == 0:                                   # update weights every 10 epochs
            max_eqa, _ = grad_stats(phys_loss, params)
            _, mean_bc   = grad_stats(boundary_loss_value, params)
            _, mean_data = grad_stats(data_loss_value, params)
            alpha = (1 - a)*alpha + a*(max_eqa/(mean_bc + 1e-12)).item()
            beta  = (1 - a)*beta  + a*(max_eqa/(mean_data + 1e-12)).item()

        # Total loss (you can weight the losses if needed)
        """TEST2: fixed the weights for the different loss components, data loss: 100, boundary loss: 10, physics loss: 1 """
        #alpha, beta = 10.0, 100.0
        total_loss = phys_loss + beta * data_loss_value + alpha * boundary_loss_value
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # validation loss: governing equations + validation data
        val_loss = phys_loss.item() + data_loss(model, x_v, y_v, ux_v).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            running_loss['best_alpha']=alpha
            running_loss['best_beta']=beta
            running_loss["best_phys"]=phys_loss.item()
            running_loss["best_data"]=data_loss_value.item()
            running_loss["best_boundary"]=boundary_loss_value.item()
            running_loss["best_total"]=total_loss.item()

        running_loss["phys"].append(phys_loss.item())
        running_loss["data"].append(data_loss_value.item()) 
        running_loss["boundary"].append(boundary_loss_value.item())
        running_loss["total"].append(total_loss.item())
        running_loss['alpha'].append(alpha)
        running_loss['beta'].append(beta)

        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], physics Loss: {phys_loss.item():.4f}, \
                   data Loss: {data_loss_value.item():.4f}, boundary Loss: {boundary_loss_value.item():.4f}, \
                    total Loss: {total_loss.item():.4f}, alpha_bc: {alpha:.2f}, beta_data: {beta:.2f}')
    # Load the best model state
    model.load_state_dict(best_state)
    return model, running_loss