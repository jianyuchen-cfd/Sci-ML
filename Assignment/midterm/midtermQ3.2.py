import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import jax, jax.numpy as jnp
from jax import random
import optax

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---------------- load sampler (from the assignment) ----------------
def sample_random_load(key):
    key_center, key_width, key_gauss, key_fourier, key_poly = random.split(key, 5)
    center = random.uniform(key_center, (), minval=0.2, maxval=0.8) * L
    width = random.uniform(key_width, (), minval=0.5, maxval=2.0)
    amp_gauss = random.uniform(key_gauss, (), minval=-8.0, maxval=8.0)
    gauss = amp_gauss * jnp.exp(-((x_grid_jnp - center) / width) ** 2)

    freqs = jnp.array([1, 2, 3, 4], dtype=jnp.float32)
    coeffs = random.uniform(key_fourier, (4,), minval=-5.0, maxval=5.0)
    angles = freqs[:, None] * jnp.pi * x_grid_jnp[None, :] / L
    sines = jnp.sum(coeffs[:, None] * jnp.sin(angles), axis=0)

    key_poly1, key_poly2 = random.split(key_poly)
    poly_coeff = random.uniform(key_poly1, (), minval=-2.0, maxval=2.0)
    slope_coeff = random.uniform(key_poly2, (), minval=-2.0, maxval=2.0)
    poly = poly_coeff * (x_grid_jnp / L - 0.5)
    slope = slope_coeff * (x_grid_jnp / L)
    return (gauss + sines + poly + slope).astype(jnp.float32)