import torch

NU_MOL = 0.056 / 5000
KAPPA, E_WALL = 0.41, 9.8

def sample_near_wall(n, a, L=3.0, H=1.0, h1=0.056, h2=0.16):
    """n extra collocation points per solid wall, within distance a of it."""
    d = a * torch.rand(4 * n, 1)
    dc, df, dl, dr = d[:n], d[n:2 * n], d[2 * n:3 * n], d[3 * n:]
    x = torch.cat([L * torch.rand(n, 1), L * torch.rand(n, 1), dl, L - dr])
    y = torch.cat([H - dc, df, (H - h1) * torch.rand(n, 1), h2 + (H - h2) * torch.rand(n, 1)])
    return x, y


def wall_zones(x, y, a, L=3.0, H=1.0, h1=0.056, h2=0.16):
    """Distance d to the nearest wall, which wall (0 ceiling, 1 floor, 2 left, 3 right)"""
    d, k = torch.cat([H - y, y, x, L - x], dim=1).min(dim=1)
    yy = y.squeeze(1)
    solid = (k < 2) | ((k == 2) & (yy < H - h1)) | ((k == 3) & (yy > h2))
    return d.unsqueeze(1), k.unsqueeze(1), (d < a) & solid

def wall_law_target(A_d, nu_eff, d, b, nu=NU_MOL, kappa=KAPPA, E=E_WALL):
    """Wall-law velocity at distance d, from the local slope A_d (>0).
    Stress balance:  u_tau^2 = nu_eff * A_d.
    d < b (sublayer): u = u_tau^2 d / nu_mol = d * A_d.      d >= b (log law): u = (u_tau/kappa) ln(E d u_tau / nu_mol)."""
    A_d = A_d.clamp_min(0.0)
    ut = torch.sqrt(nu_eff * A_d)
    u_log = ut / kappa * torch.log((E * d * ut / nu).clamp_min(1.0))
    u_sub = d * A_d
    return torch.where(d < b, u_sub, u_log)
 
 
def wall_function_loss(model, x, y, d, k, b=0.003, nu=NU_MOL, c_nut=0.03874):
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    psi = model(torch.cat([x, y], dim=1))[:, 0:1]
    u, v = grad(psi, y), -grad(psi, x)
    u_y, v_x = grad(u, y), grad(v, x)
    t = torch.where(k < 2, u, v)                                                                # velocity along the wall
    t_d = torch.where(k == 0, -u_y, torch.where(k == 1, u_y, torch.where(k == 2, v_x, -v_x)))   # derivative away from the wall
    s = torch.sign(t).detach()
    A, A_d = s * t, (s * t_d).detach()                                                          # u_tau is frozen (detached)
    nu_eff = (nu + c_nut * torch.sqrt(u ** 2 + v ** 2 + 1e-8) * d).detach()                     # same viscosity as in the momentum equation
    target = wall_law_target(A_d, nu_eff, d, b, nu)
    valid = (A_d > 1e-3).float()                                                                # only where the velocity grows away from the wall
    R = (A - target) / (A.detach().abs() + 0.1)
    return torch.sum(valid * R ** 2) / valid.sum().clamp_min(1.0)

def physic_loss(model, x, y, Re, nu_mol=1.12e-5, c_nut=0.03874, L=3, H=1):
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

def physic_loss_zeroeq(model, x, y, nu_mol=0.056 / 5000, c_nut=0.03874, L=3.0, H=1.0):
    """Momentum residual with eddy viscosity nu_t = c_nut*|u|*d (d = distance to the nearest wall); nu_eff = nu_mol + nu_t."""
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    out = model(torch.cat([x, y], dim=1))
    psi, p = out[:, 0:1], out[:, 1:2]
    u, v = grad(psi, y), -grad(psi, x)
    d = torch.minimum(torch.minimum(x, L - x), torch.minimum(y, H - y))
    nu = nu_mol + c_nut * torch.sqrt(u ** 2 + v ** 2 + 1e-8) * d
    u_x, u_y, v_x, v_y = grad(u, x), grad(u, y), grad(v, x), grad(v, y)
    p_x, p_y = grad(p, x), grad(p, y)
    txx, txy, tyy = 2 * nu * u_x, nu * (u_y + v_x), 2 * nu * v_y
    f_x = u * u_x + v * u_y + p_x - (grad(txx, x) + grad(txy, y))
    f_y = u * v_x + v * v_y + p_y - (grad(txy, x) + grad(tyy, y))
    return torch.mean(f_x ** 2 + f_y ** 2)

def grad(f,v):
    return torch.autograd.grad(f, v, grad_outputs=torch.ones_like(f), create_graph=True)[0]

def inlet_loss(model, n=20, u_in=1.0, H=1.0, h1=0.056):
    # inlet slot: x = 0, 1-h1 < y < 1 ; uniform u_x = u_in, u_y = 0
    x = torch.zeros(n, 1, requires_grad=True)
    y = ((H - h1) + h1 * torch.rand(n, 1)).requires_grad_(True)
    psi = model(torch.cat([x, y], dim=1))[:, 0:1]
    u, v = grad(psi, y), -grad(psi, x)
    return torch.mean((u - u_in) ** 2 + v ** 2)

def outlet_loss(model, n=40, L=3.0, h2=0.16):
    # outlet slot: x = L, 0 < y < h2 ; zero normal gradient (Neumann): du/dx = 0, dv/dx = 0
    x = torch.full((n, 1), L, requires_grad=True)
    y = (h2 * torch.rand(n, 1)).requires_grad_(True)
    psi = model(torch.cat([x, y], dim=1))[:, 0:1]
    u, v = grad(psi, y), -grad(psi, x)
    return torch.mean(grad(u, x) ** 2 + grad(v, x) ** 2)

def boundary_loss(model, x_boundary, y_boundary, H=1, L=3, relax_horizontal=False):
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
        horiz = ((y_boundary < 1e-6) | (y_boundary > H - 1e-6)).float()
        vert  = ((x_boundary < 1e-6) | (x_boundary > L - 1e-6)).float()
        loss = torch.mean(u**2 * (1 - horiz) + v**2 * (1 - vert))
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