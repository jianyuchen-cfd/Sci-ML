#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 10:18:43 2026

@author: jianyuch
"""

import numpy as np
import matplotlib.pyplot as plt


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