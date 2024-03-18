import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import seaborn as sns
import os
from tqdm import tqdm      

try:
    os.mkdir('results_final')
except:
    pass

data = []
with open("startend.dat","r") as f:
    data = f.readlines()
data=[i.strip().replace('\n','').strip().split(" ") for i in data]
N = int(data[0][0])
nt = int(data[0][1])
nhalf = int(data[0][4])
method = int(data[0][5])
nthreads = int(data[0][6])
nprocs = int(data[0][7])
d = {1:"lax",2:"first",3:"second"}
method = d[method]
data.pop(0)
data = [[float(i) for i in j] for j in data]

start = data[:N]
data = data[N:]
mid = data[:N]
end = data[N:]

plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.title("N = %d nprocs = %d nthreads = %d\n on timestep %d out of %d"%(N,nprocs,nthreads,0,nt))
plt.imshow(np.rot90(start,1), cmap='hot', interpolation='none', vmin=0.0, vmax=1.0, extent=[-1,1,-1,1])
plt.colorbar()
plt.savefig(f"results_final/{str(N)},{str(nt)}-{nprocs},{nthreads}-start.png")
plt.clf()

plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.title("N = %d nprocs = %d nthreads = %d\n on timestep %d out of %d"%(N,nprocs,nthreads,nhalf,nt))
plt.imshow(np.rot90(mid,1), cmap='hot', interpolation='none', vmin=0.0, vmax=1.0, extent=[-1,1,-1,1])
plt.colorbar()
plt.savefig(f"results_final/{str(N)},{str(nt)}-{nprocs},{nthreads}-mid.png")
plt.clf()


plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.title("N = %d nprocs = %d nthreads = %d\n on timestep %d out of %d"%(N,nprocs,nthreads,nt,nt))
plt.imshow(np.rot90(end,1), cmap='hot', interpolation='none', vmin=0.0, vmax=1.0, extent=[-1,1,-1,1])
plt.colorbar()
plt.savefig(f"results_final/{str(N)},{str(nt)}-{nprocs},{nthreads}-end.png")
plt.clf()

print("Plots saved in results_final directory")
print(f"{str(N)},{str(nt)}-{nprocs},{nthreads}-xxx.png")