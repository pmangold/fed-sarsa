import pickle
import numpy as np

import os


import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams['text.usetex'] = True

S=10


import matplotlib.pyplot as plt
import seaborn as sns
palette = sns.color_palette("colorblind")
# palette = [palette_[0], palette_[1], palette_[4]]

num_points = 1000
max_plot = 200
alpha = 0.01

# plt.rcParams.update({
#     'legend.fontsize': 18,             
#     'font.family': 'lmodern',
#     'text.usetex': True
# })

import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{mathptmx}",
    "font.size": 20,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


markers = ["o", "+", "^", "x"]


os.makedirs("plots/S_" + str(S), exist_ok=True)


# 100000
Ts = [0, 20, 40, 60, 80, -1]
print("T,0,2000,4000,6000,8000,10000")

# plot the legend
handles = [plt.plot([],[],marker=markers[0], color=palette[0], ls="solid")[0],
           plt.plot([],[],marker=markers[1], color=palette[1], ls="solid")[0],
           plt.plot([],[],marker=markers[2], color=palette[2], ls="solid")[0],
           plt.plot([],[],color="black", ls="dashed")[0],
           plt.plot([],[],color="black", ls="solid")[0]
           ]
labels = ["H=1", "H=100", "H=10000", "MSE (average env.)", "MSE ($\\theta_\\star$)"]
legend = plt.legend(handles, labels, frameon=False, ncol=5)

plt.axis('off')
fig  = legend.figure
bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig("plots/S_" + str(S) + "/legend.pdf", dpi="figure", bbox_inches=bbox)


steps = 1_000_000
num_logs = 1_000 #MSE_fed.shape[1]  # should match your logged data
log_times = np.round(np.linspace(1, steps, num_logs)).astype(int)

for N in [2, 10, 20, 50]: #[2, 10, 20, 50, 100, 200]: #, 10, 20, 50, 100, 200]:
    with open("results/S_" + str(S) + "/"  + str(N) + "_" + str(alpha) + "/expe_local_step.pickle", "rb") as f:
        data = pickle.load(f)

    theta_lim_avg = data["theta_lim_avg"]
    theta_lim_fed = data["theta_lim_fed"]
    seeds = data["seeds"]
    H_list = data["H_list"]

    print(data)




    fig, ax = plt.subplots(1, 1, figsize=(4,3))
    
    for i, H in enumerate(H_list):

        print(H, end="")

        
        thetas = data["thetas"][H]


        MSE_fed = np.linalg.norm(thetas - theta_lim_fed, axis=2)**2
        MSE_avg = np.linalg.norm(thetas - theta_lim_avg, axis=2)**2

        idx = np.linspace(0, len(MSE_fed[0])-1, num_points, dtype=int)

        xidx =  np.linspace(0, 1_000_000, len(MSE_fed[0]), dtype=int) # np.arange(len(MSE_fed[0]))
        
        print("len", len(idx), len(MSE_fed[0]))
        
        print(idx)

        std_fed = np.std(MSE_fed[:, idx], axis=0)
        mean_fed = np.mean(MSE_fed[:, idx], axis=0)
        min_fed = np.min(MSE_fed[:, idx], axis=0)
        
        std_avg = np.std(MSE_avg[:, idx], axis=0)
        mean_avg = np.mean(MSE_avg[:, idx], axis=0)
        min_avg = np.min(MSE_avg[:, idx], axis=0)

        plt.fill_between(xidx[:max_plot],
                         np.maximum(mean_fed - std_fed, min_fed)[:max_plot],
                         (mean_fed + std_fed)[:max_plot],
                         color=palette[i], alpha=0.3)
        plt.fill_between(xidx[:max_plot],
                         np.maximum(mean_avg - std_avg, min_avg)[:max_plot],
                         (mean_avg + std_avg)[:max_plot],
                         color=palette[i], alpha=0.3)


        for T in Ts:
            print(",", end="")
            print("%.2f" % mean_fed[T], end="")
        print()
        
        plt.plot(xidx[:max_plot], mean_fed[:max_plot], color=palette[i], marker=markers[i], markevery=20, label="fed, H="+str(H))
        plt.plot(xidx[:max_plot], mean_avg[:max_plot], color=palette[i], marker=markers[i], markevery=20, linestyle="dashed", label="avg, H="+str(H))
        


    plt.ylim(2e-5, 8e1)
    # plt.xticks([1, 50000, 100000], ["1", "5e4", "10e4"])
    
    plt.yscale("log")

    plt.savefig("plots/S_" + str(S) + "/" + str(N) + ".pdf", bbox_inches="tight")


