#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hodkin-Huxley Neuron Model

Created on Wed Apr  4 21:17:56 2018
@author: shinestar
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

class Neuron():
    def __init__(self):
        # Set random seed (for reproducibility)
        np.random.seed(1000)
        
        # Start and end time (in milliseconds)
        self.tmin = 0.0
        self.tmax = 50.0
        
        # Average potassium channel conductance per unit area (mS/cm^2)
        self.gK = 36.0
        
        # Average sodoum channel conductance per unit area (mS/cm^2)
        self.gNa = 120.0
        
        # Average leak channel conductance per unit area (mS/cm^2)
        self.gL = 0.3
        
        # Membrane capacitance per unit area (uF/cm^2)
        self.Cm = 1.0
        
        # Potassium potential (mV)
        self.VK = -12.0
        
        # Sodium potential (mV)
        self.VNa = 115.0
        
        # Leak potential (mV)
        self.Vl = 10.613
        
        # Time values
        self.T = np.linspace(self.tmin, self.tmax, 10000)
    
    # Potassium ion-channel rate functions
    
    def alpha_n(self, Vm):
        return (0.01 * (10.0 - Vm)) / (np.exp(1.0 - (0.1 * Vm)) - 1.0)
    
    def beta_n(self, Vm):
        return 0.125 * np.exp(-Vm / 80.0)
    
    # Sodium ion-channel rate functions
    
    def alpha_m(self, Vm):
        return (0.1 * (25.0 - Vm)) / (np.exp(2.5 - (0.1 * Vm)) - 1.0)
    
    def beta_m(self, Vm):
        return 4.0 * np.exp(-Vm / 18.0)
    
    def alpha_h(self, Vm):
        return 0.07 * np.exp(-Vm / 20.0)
    
    def beta_h(self, Vm):
        return 1.0 / (np.exp(3.0 - (0.1 * Vm)) + 1.0)
      
    # n, m, and h steady-state values
    
    def n_inf(self, Vm=0.0):
        return self.alpha_n(Vm) / (self.alpha_n(Vm) + self.beta_n(Vm))
    
    def m_inf(self, Vm=0.0):
        return self.alpha_m(Vm) / (self.alpha_m(Vm) + self.beta_m(Vm))
    
    def h_inf(self, Vm=0.0):
        return self.alpha_h(Vm) / (self.alpha_h(Vm) + self.beta_h(Vm))
      
    # Input stimulus
    def Id(self, t):
        if 0.0 < t < 1.0:
            return 150.0
        elif 10.0 < t < 11.0:
            return 50.0
        return 0.0
      
    # Compute derivatives
    def compute_derivatives(self, y, t0):
        dy = np.zeros((4,))
        
        Vm = y[0]
        n = y[1]
        m = y[2]
        h = y[3]
        
        # dVm/dt
        GK = (self.gK / self.Cm) * np.power(n, 4.0)
        GNa = (self.gNa / self.Cm) * np.power(m, 3.0) * h
        GL = self.gL / self.Cm
        
        dy[0] = (self.Id(t0) / self.Cm) - (GK * (Vm - self.VK)) - (GNa * (Vm - self.VNa)) - (GL * (Vm - self.Vl))
        
        # dn/dt
        dy[1] = (self.alpha_n(Vm) * (1.0 - n)) - (self.beta_n(Vm) * n)
        
        # dm/dt
        dy[2] = (self.alpha_m(Vm) * (1.0 - m)) - (self.beta_m(Vm) * m)
        
        # dh/dt
        dy[3] = (self.alpha_h(Vm) * (1.0 - h)) - (self.beta_h(Vm) * h)
        
        return dy


# State (Vm, n, m, h)
        
HH = Neuron()
Y = np.array([0.0, HH.n_inf(), HH.m_inf(), HH.h_inf()])

# Solve ODE system
# Vy = (Vm[t0:tmax], n[t0:tmax], m[t0:tmax], h[t0:tmax])
Vy = odeint(HH.compute_derivatives, Y, HH.T)

# Input stimulus
Idv = [HH.Id(t) for t in HH.T]

fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(HH.T, Idv)
ax.set_xlabel('Time (ms)')
ax.set_ylabel(r'Current density (uA/$cm^2$)')
ax.set_title('Stimulus (Current density)')
plt.grid()

# Neuron potential
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(HH.T, Vy[:, 0])
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Vm (mV)')
ax.set_title('Neuron potential with two spikes')
plt.grid()

# Trajectories with limit cycles
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(Vy[:, 0], Vy[:, 1], label='Vm - n')
ax.plot(Vy[:, 0], Vy[:, 2], label='Vm - m')
ax.set_title('Limit cycles')
ax.legend()
plt.grid()
