import copy
import torch
from Calculate_Loss import data_loss, physic_loss, boundary_loss



def sample_collocation(n, L, H):
    x = L * torch.rand(n, 1)
    y = H * torch.rand(n, 1)
    return x, y

def train_model(model, optimizer,  boundary_points, data_points, val_points, Re,
             n_collocation_points=640, L=3.0, H=1.0, num_epochs=3000):
    """
    Train the PINN model using physics-informed loss and boundary condition loss.

    Parameters:
    - model: The neural network model (PINN).
    - optimizer: The optimizer for training.
    - val_points: The validation points for evaluating the model.
    - collocation_points: The collocation points for the physics-informed loss.
    - boundary_points: The boundary points for the boundary condition loss.
    - data_points: The data points for the data loss.
    - Re: Reynolds number
    - num_epochs: Number of training epochs

    Returns:
    - model: The trained model.
    """
    x_b, y_b = boundary_points
    x_d, y_d, ux_d = data_points
    x_v, y_v, ux_v = val_points

    best_val, best_state = float('inf'), None
    running_loss = {"phys": [], "data": [], "boundary": [], "total": []}

    def grad_stats(loss, params):
        g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        g = torch.cat([gi.flatten() for gi in g if gi is not None])
        return g.abs().max(), g.abs().mean()

    alpha, beta, a = 1.0, 1.0, 0.9
    params = [p for p in model.parameters()]

    for epoch in range(num_epochs):
        x_c, y_c = sample_collocation(n=n_collocation_points, L=L, H=H)  
        # Calculate physics-informed loss
        phys_loss = physic_loss(model, x_c, y_c, Re)
        # Calculate data loss (assuming you have data)
        data_loss_value = data_loss(model, x_d, y_d, ux_d)

        # Calculate boundary condition loss (assuming you have boundary data)
        boundary_loss_value = boundary_loss(model, x_b, y_b)
        if epoch % 10 == 0:                                   # update weights every 10 epochs
            max_eqa, _ = grad_stats(phys_loss, params)
            _, mean_bc   = grad_stats(boundary_loss_value, params)
            _, mean_data = grad_stats(data_loss_value, params)
            alpha = (1 - a)*alpha + a*(max_eqa/(mean_bc + 1e-12)).item()
            beta  = (1 - a)*beta  + a*(max_eqa/(mean_data + 1e-12)).item()

        # Total loss (you can weight the losses if needed)
        total_loss = phys_loss + beta * data_loss_value + alpha * boundary_loss_value
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # validation loss: governing equations + validation data
        val_loss = phys_loss.item() + data_loss(model, x_v, y_v, ux_v).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

        running_loss["phys"].append(phys_loss.item())
        running_loss["data"].append(data_loss_value.item()) 
        running_loss["boundary"].append(boundary_loss_value.item())
        running_loss["total"].append(total_loss.item())
        running_loss['alpha'] = alpha
        running_loss['beta'] = beta

        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], physics Loss: {phys_loss.item():.4f}, \
                   data Loss: {data_loss_value.item():.4f}, boundary Loss: {boundary_loss_value.item():.4f}, \
                    total Loss: {total_loss.item():.4f}, alpha: {alpha:.2f}, beta: {beta:.2f}')
    # Load the best model state
    model.load_state_dict(best_state)
    return model, running_loss