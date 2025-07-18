import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
import sys
import pickle
import pandas as pd

# Make mean of mulitple files
filename = []
for i, val in enumerate(sys.argv):
    if i >= 1:    
        filename.append(val)

# Load csvs
for data_csv in filename:
    data = pd.read_csv(data_csv)

    res_mean, res_std = data['res_mean'][:600], data['res_std'][:600]
    opt_gain_mean, opt_gain_std = data['opt_gain_mean'][:600], data['opt_gain_std'][:600]
    opt_res_mean, opt_res_std = data['opt_res_mean'][:600], data['opt_res_std'][:600]

    labels = data_csv.split("_")
    if len(labels) > 2: # AAFIB result
        safeguard = labels[1] + "_" + labels[2] + "_" + labels[3]
        env = labels[-1].split(".")[0]
    else:
        safeguard = labels[0]
        env = labels[-1].split(".")[0]
    
    plt.plot(np.log10(res_mean), label=safeguard)
    # plt.fill_between(np.linspace(0,data.shape[0],data.shape[0]), data+data_std, data-data_std, alpha=0.15)

plt.legend()
plt.title(env)
plt.xlabel('iter (k)')
plt.ylabel(r'$\log(\|g^k\|_\infty$)')
# plt.xlim(0,200)
# plt.xticks(np.arange(0,601,100))
# plt.ylim(-6, 2)
# plt.yticks(np.arange(-6,3,2))

plt.grid()
plt.show()
plt.savefig('./{}.png'.format(env))
plt.clf()
