import numpy as np
import matplotlib.pyplot as plt
from neuron import sim_neurons
#IUAON/CHA0STHE0RX
N_neurons = 2
T = 300              # Total time 
dt = 0.01            # Time step
time_steps = int(T / dt)
#W = np.random.uniform(0.05, 0.1, (N_neurons, N_neurons))
W=np.zeros((N_neurons,N_neurons))
W[1,0]=0.05
np.fill_diagonal(W, 0.0)
Ie_in = np.zeros((time_steps, N_neurons))
Ie_in[int(20/dt):int(25/dt),0] = 12 #pre training bell
for p in [80, 120, 160]:
    Ie_in[int(p/dt):int((p+5)/dt),0] = 12
    Ie_in[int((p+2)/dt):int((p+7)/dt),1] = 12
Ie_in[int(245/dt):int(250/dt),0] = 12 #post training bell

time,V_out,W_out= sim_neurons(Ie_in,W,T,dt)
plt.figure(figsize=(10, 6))

for n in range(N_neurons):
    spike_times = np.where((V_out[1:, n] > 0.0) & (V_out[:-1, n] <= 0.0))[0]
    spike_times = spike_times * dt
    plt.scatter(spike_times, [n]*len(spike_times), color='black', s=10) #s=10 is the dot size

plt.axvline(x=50, color='red', linestyle='--', alpha=0.5, label="Start Training")
plt.axvline(x=200, color='green', linestyle='--', alpha=0.5, label="End Training")
plt.title("Chain firing")
plt.xlabel("Time (ms)")
plt.ylabel("Neuron ID")
plt.ylim(-1, N_neurons)
plt.yticks([0,1],["0-Bell","1-Dog"])
plt.xlim(0, T)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
print(f"Starting Weight: 0.050")
print(f"Final Weight:    {W_out[1, 0]:.3f}")