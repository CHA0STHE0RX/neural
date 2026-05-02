import numpy as np
import matplotlib.pyplot as plt
#from numba import jit
#IUAON/CHA0STHE0RX
def sim_neurons(Ie_mx,W,T,dt,stdp=True,inhs=None): #current array,weight matrix, max time, time step
    
    C = 1.0      # Membrane capacitance
    G_Na = 120.0 # Max Sodium conductance
    G_K = 36.0   # Max Potassium conductance
    G_L = 0.3    # Max Leak conductance
    E_Na = 50.0  # Sodium reversal potential
    E_K = -77.0  # Potassium reversal potential
    V_L = -54.4  # Leak reversal potential
    alpha_s = 2.0 # Synapse binding rate
    beta_s = 0.2 # Synapse unbinding rate
    E_syn = 0.0 # Synaptic reversal potential
    alpha_m = lambda V: 0.1*(V+40)/(1-np.exp(-(V+40.001)/10)) 
    beta_m = lambda V: 4*np.exp(-(V+65.001)*(139/2500))
    alpha_h = lambda V: 0.07*np.exp(-0.05*(V+65.001))
    beta_h = lambda V: 1/(1+np.exp(-0.1*(V+35.001)))
    alpha_n = lambda V: 0.01*(V+55.001)/(1-np.exp(-0.1*(V+55.001)))
    beta_n = lambda V: 0.125*np.exp(-0.0125*(V+65.001))

    # STDP Constants
    A_plus = 0.01   #LTP
    A_minus = 0.012 # LTD
    tau = 20.0      # Time Window (20 ms)

    N_neurons=Ie_mx.shape[1]
    last_spike = np.full(N_neurons, -1000.0) 
    s= np.full(N_neurons,0.0) #Synaptic state
    s_exc=np.full(N_neurons,0.0) #Excitatory synaptic state
    s_inh=np.full(N_neurons,0.0) #Inhibitory synaptic state
    t=np.arange(0,T,dt)
    t_steps=len(t)
    V = np.zeros((t_steps,N_neurons))
    V[0,:]=-65   # Initial voltage
    #Gating Variables
    m= np.full(N_neurons,0.05) #Sodium activation
    h= np.full(N_neurons,0.6) #Sodium inactivation
    n= np.full(N_neurons,0.32) #Potassium activation
    # n0->n1 with weight of 0.8 W[1, 0] = 0.8

    for i in range(1,len(t)):
        Ie = Ie_mx[i,:]  #external current
        lV= V[i-1,:] #last voltage
        T_nt = (lV > 0.0).astype(float)
        ds= (alpha_s*T_nt*(1-s)-beta_s*s)*dt
        s= np.clip(s+ds,0.0,1.0)
        if inhs is not None:
            s_exc=s*(~inhs)
            s_inh=s*(inhs)
            I_exc=W.dot(s_exc)*(lV-E_syn)
            I_inh=W.dot(s_inh)*(lV-E_K)
            I_syn=(I_exc+I_inh)
        else:
            I_syn=W.dot(s)*(lV-E_syn)
        dV=(-G_L*(lV-V_L) -G_Na*m**3*h*(lV-E_Na) - G_K*n**4*(lV-E_K)+Ie-I_syn)*dt/C #Master equation
        V[i,:]=lV+dV + np.random.normal(0,0.05,size=N_neurons)
        current_time = i*dt
        if stdp:
            just_spiked = (V[i-1, :] > 0.0) & (V[i-2, :] <= 0.0)
            for j in range(N_neurons):
                if just_spiked[j]:
                    delta_t = current_time - last_spike  # Array of times
                    #Ignore non-firing neurons
                    valid = (delta_t > 0) & (delta_t<100)
                    # 1. LTP
                    # Increase weights to j from neurons that fired recently
                    W[j, valid] += A_plus * np.exp(-delta_t[valid] / tau)
                    # 2.LTD
                    W[valid, j] -= A_minus * np.exp(-delta_t[valid] / tau)
                    last_spike[j] = current_time
                    W = np.clip(W, 0.0, 2.0)
        dm=(alpha_m(lV)*(1-m)-beta_m(lV)*m)*dt
        dh=(alpha_h(lV)*(1-h)-beta_h(lV)*h)*dt
        dn=(alpha_n(lV)*(1-n)-beta_n(lV)*n)*dt
        m = np.clip(m + dm, 0.0, 1.0); h = np.clip(h + dh, 0.0, 1.0); n = np.clip(n + dn, 0.0, 1.0)
    return t,V,W
