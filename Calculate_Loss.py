import torch

def physic_loss(model, x, y, Re):
    """
    Calculate the physics-informed loss for a Navier-Stokes momentum equation.

    Parameters:
    - model: The neural network model (PINN).
    - x: Input tensor for spatial x direction
    - y: Input tensor for spatial y direction

    Returns:
    - loss: The computed physics-informed loss.
    """
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    input_tensor = torch.cat((x, y), dim=1)  # Concatenate x and y to form the input tensor

    # Forward pass through the model
    output = model(input_tensor)
    psi, p = output[:, 0:1], output[:, 1:2]  # Assuming the model outputs stream function and pressure

    u = grad(psi, y)  # u = d(psi)/dy
    v = -grad(psi, x)  # v = -d(psi)/dx

    u_x, u_y = grad(u, x), grad(u, y)
    v_x, v_y = grad(v, x), grad(v, y)
    p_x, p_y = grad(p, x), grad(p, y)

    u_xx, u_yy = grad(u_x, x), grad(u_y, y)
    v_xx, v_yy = grad(v_x, x), grad(v_y, y)

    f_x = u * u_x + v * u_y + p_x - (u_xx + u_yy) / Re  # x-momentum equation residual
    f_y = u * v_x + v * v_y + p_y - (v_xx + v_yy) / Re  # y-momentum equation residual

    residuals = torch.mean(f_x ** 2 + f_y ** 2)  # Replace this with the actual computation of residuals

    return residuals

def grad(f,v):
    return torch.autograd.grad(f, v, grad_outputs=torch.ones_like(f), create_graph=True)[0]

def boundary_loss(model, x_boundary, y_boundary):
    """
    Calculate the boundary condition loss for the stream function.

    Parameters:
    - model: The neural network model (PINN).
    - x_boundary: Input tensor for spatial x direction at the boundary
    - y_boundary: Input tensor for spatial y direction at the boundary
    - psi_boundary: The known stream function values at the boundary

    Returns:
    - loss: The computed boundary condition loss.
    """
    x_boundary = x_boundary.clone().detach().requires_grad_(True)
    y_boundary = y_boundary.clone().detach().requires_grad_(True)
    input_tensor = torch.cat((x_boundary, y_boundary), dim=1)  # Concatenate x and y to form the input tensor

    # Forward pass through the model
    output = model(input_tensor)
    psi_pred = output[:, 0:1]  # Assuming the model outputs stream function and pressure

    u, v = grad(psi_pred, y_boundary), -grad(psi_pred, x_boundary)  # Compute velocity components from stream function
    
    # Compute the mean squared error between predicted and known stream function values
    loss = torch.mean((u - 0) ** 2 + (v - 0) ** 2)

    return loss

def data_loss(model, x_data, y_data, ux_data):
    """
    Calculate the data loss for the stream function.

    Parameters:
    - model: The neural network model (PINN).
    - x_data: Input tensor for spatial x direction at the data points
    - y_data: Input tensor for spatial y direction at the data points
    - ux_data: The known x-velocity values at the data points

    Returns:
    - loss: The computed data loss.
    """
    x_data = x_data.clone().detach().requires_grad_(True)
    y_data = y_data.clone().detach().requires_grad_(True)
    input_tensor = torch.cat((x_data, y_data), dim=1)  # Concatenate x and y to form the input tensor

    # Forward pass through the model
    output = model(input_tensor)
    psi_pred = output[:, 0:1]  # Assuming the model outputs stream function and pressure
    ux = grad(psi_pred, y_data)  # Compute x-velocity from stream function
    # Compute the mean squared error between predicted and known x-velocity values
    loss = torch.mean((ux - ux_data) ** 2)

    return loss