import torch

def wall_function_loss(model, utau_net, nu_mol=1.12e-5, n=40, y_p=0.028, x_max=2.4,
                       kappa=0.41, E=9.8, nu_t_max=0.01):
    x = (x_max * torch.rand(n, 1)).requires_grad_(True)
    y = torch.full((n, 1), 1 - y_p).requires_grad_(True)
    out = model(torch.cat([x, y], dim=1))
    u = grad(out[:, 0:1], y)
    nu_eff = nu_mol + nu_t_max * torch.sigmoid(out[:, 2:3])
    ut = utau_net(x)
    R1 = u - ut / kappa * torch.log(E * y_p * ut / nu_mol)       # log law
    R2 = (-nu_eff * grad(u, y)) / ut**2 - 1.0                    # shear balance, dimensionless
    return torch.mean(R1**2 + R2**2)

def physic_loss(model, x, y, Re, nu_mol=1.12e-5, nu_t_max=0.01):
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

    #u_xx, u_yy = grad(u_x, x), grad(u_y, y)
    #v_xx, v_yy = grad(v_x, x), grad(v_y, y)
    
    
    #f_x = u * u_x + v * u_y + p_x - (u_xx + u_yy) / Re  # x-momentum equation residual
    #f_y = u * v_x + v * v_y + p_y - (v_xx + v_yy) / Re  # y-momentum equation residual
    
    
    # Implement the wall function and effective viscosity
    s = output[:, 2:3]
    nu = nu_mol + nu_t_max * torch.sigmoid(s)
    txx, txy, tyy = 2*nu*u_x, nu*(u_y + v_x), 2*nu*v_y 
    f_x = u*u_x + v*u_y + p_x - (grad(txx, x) + grad(txy, y))
    f_y = u*v_x + v*v_y + p_y - (grad(txy, x) + grad(tyy, y))

    residuals = torch.mean(f_x ** 2 + f_y ** 2)  # Replace this with the actual computation of residuals

    return residuals

def grad(f,v):
    return torch.autograd.grad(f, v, grad_outputs=torch.ones_like(f), create_graph=True)[0]

def boundary_loss(model, x_boundary, y_boundary, relax_horizontal=False):
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
    if relax_horizontal:
        # ceiling (y=1) and floor (y=0): keep only no-penetration (v = 0), leave the tangential u free
        horiz = ((y_boundary < 1e-6) | (y_boundary > 1 - 1e-6)).float()
        loss = torch.mean(u**2 * (1 - horiz) + v**2)
    else:
        loss = torch.mean(u**2 + v**2)

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