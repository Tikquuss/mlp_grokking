"""
pip install folium==0.2.1
pip install pdflatex
sudo apt-get install texlive-latex-recommended 
sudo apt install texlive-latex-extra
sudo apt install dvipng
"""
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable
import seaborn as sns 

SAVE_IN_PGF=False

if SAVE_IN_PGF :
    try : import pdflatex
    except ModuleNotFoundError : SAVE_IN_PGF=False

if SAVE_IN_PGF :
    import matplotlib
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': False,
        #'text.usetex': True,
        'pgf.rcfonts': False,
    })
    try : matplotlib.rcParams['text.latex.unicode']=False
    except : pass

def linspace(v : list, N : int):
    M = len(v)
    N = min(N, M)
    #print(v)
    step = M if N == 1 else (M-1)//(N-1)
    tmp = v[0::step]
    tmp = v[:N] if len(tmp) == len(v) else tmp[:N]
    return tmp

def display_pca_scatterplot_simple(data_dim, words, N, save_path, filename, plot_line = True, cmap='viridis', eps = 0.0, 
                                   preicted_data_dim = None, legend_loc="center") :
    data = {"w" : [], "x" : [], "y" : []}
    for k, w in enumerate(words) :
        i, j = data_dim[k]
        data["w"].append(w)
        data["x"].append(i.item())
        data["y"].append(j.item())
    data = pd.DataFrame.from_dict(data)
    #df.head(4)

    #sns.set(style="whitegrid")
    sns.set_theme(style="white", palette=None)

    H, L = 1, 1
    fig, axs = plt.subplots(H, L, figsize = (L*7, H*6), sharex=False, sharey=False) 
    
    viridis = plt.get_cmap(cmap)
    colors = linspace(v = viridis.colors, N = N)
    # while N > len(colors) :
    #     colors += colors
    viridis = ListedColormap(colors)
    #plt.scatter(x = data["x"], y = data["y"], s=None, c=None, marker=None, cmap=viridis, norm=None)
    sns.scatterplot(ax = axs, x = data["x"], y = data["y"], hue=data["w"], 
                    #palette=colors, 
                    palette=cmap if N > len(colors) else colors, 
                    #cmap="viridis",
                    #cmap=viridis,
                    legend = False, 
                    label= None if plot_line else ("empirical" if preicted_data_dim is not None else None)
                    )

    if plot_line : 
        colors2 = linspace(v = plt.get_cmap(cmap).colors, N = N)
        axs.plot(data["x"], data["y"],
                 color=tuple(colors2[N//2]),
                 label= ("empirical" if preicted_data_dim is not None else None) if plot_line else None
                 )

    # add annotations one by one with a loop
    for line in range(0,data.shape[0]):
        axs.text(data.x[line]+eps, data.y[line], words[line], 
                  #horizontalalignment='left', 
                  #size='medium', 
                  #color='black', 
                  #weight='semibold',
                  va='center'
                  )

    if preicted_data_dim is not None :
        data = {"w" : [], "x" : [], "y" : []}
        for k, w in enumerate(words) :
            i, j = preicted_data_dim[k]
            data["w"].append(w)
            data["x"].append(i.item())
            data["y"].append(j.item())
        data = pd.DataFrame.from_dict(data)

        sns.scatterplot(ax = axs, x = data["x"], y = data["y"], hue=data["w"], 
                        palette=cmap if N > len(colors) else colors,  ###################### 
                        legend = False, marker='*', 
                        label= None if plot_line else ("predicted" if preicted_data_dim is not None else None)
                        )
        #colors2 = linspace(v = plt.get_cmap(cmap).colors, N = N)
        #axs.plot(data["x"], data["y"], '--', color=tuple(colors2[N//2]))
        if plot_line : axs.plot(data["x"], data["y"], '--', color=tuple(colors2[N//2]), 
                               label= ("predicted" if preicted_data_dim is not None else None) if plot_line else None
                        )
        #for line in range(0,data.shape[0]): axs.text(data.x[line]+eps, data.y[line], words[line], va='center')

    axs.set_xlabel("")
    axs.set_ylabel("")
    axs.legend(loc=legend_loc, frameon=False)

    axs.spines["right"].set_visible(False)
    axs.spines["top"].set_visible(False)
    axs.axis("off")

    sns.set()
    #plt.show()
    plt.savefig(f'{save_path}/{filename}.png')
    if SAVE_IN_PGF:plt.savefig(f'{save_path}/{filename}.pgf')


@torch.no_grad()
def pairwise_distances(mat : torch.Tensor) -> torch.Tensor:
    """
    Computes the distance between pairs of elements of a matrix in a tensorized way : (x - y)^2 = x^2 - 2*x*y + y^2
    https://discuss.pytorch.org/t/build-your-own-loss-function-in-pytorch/235/7
    Parameters:
        * mat : torch.Tensor(n, dim), 2d data matrix, samples on rows and features on columns.
    Returns:
        * d : torch.Tensor(n, n), d[i][j]^2 = (mat[i] - mat[j])^2 =  mat[i]^2 - 2*mat[i]*mat[j] + mat[j]^2
    """
    # get the product x * y with y = x.t()
    r = torch.mm(mat, mat.t()) # (n, n)
    # get the diagonal elements
    diag = r.diag().unsqueeze(0).expand_as(r) # (n, n)
    # compute the distance matrix
    distances = diag + diag.t() - 2*r # (n, n)
    return distances.sqrt() # (n, n)


def analyze(model, option, save_path = ""):
    assert option in [1, 2, 3, 4]

    E = model.embeddings.weight # n x dim

    pw_dist = pairwise_distances(E) # (n, n)
    N = E.size(0)
    pw_dist = pw_dist.flatten()[1:].view(N-1, N+1)[:,:-1].reshape(N, N-1) # (n, n-1), remove the diagonal (0...)
    k=2
    tk = pw_dist.topk(k=k,dim=1,largest=False,sorted=True)
    T = tk.values # (n, k)

    a = torch.arange(model.hparams.p-3+1)
    b = a + 2
    v = torch.stack([a, b], dim=1)
    v1, v2 = v.unbind(dim=1) # (n-2, k), (n-2, k)

    v_hat = tk.indices[1:-1]
    v1_hat, v2_hat = v_hat.unbind(dim=1) # (n-2, k), (n-2, k)

    x = torch.arange(1, N-1)

    cmap='viridis'

    if option == 1 :
        y = torch.zeros_like(x)
        for i in range(N-2) :
            tmp = v[i].tolist()
            tmp2 = v_hat[i].tolist()
            #print(tmp, tmp2)
            s = 0
            for j in tmp2 :
                if j in tmp : s += 1
            y[i] = s

        # plt.figure(figsize=(5,5))    
        # plt.scatter(x,y, marker='o')
        # plt.show()

        data = {"x" : x, "y" : y}
        data = pd.DataFrame.from_dict(data)

        sns.set(style="whitegrid")
        #sns.set_theme(style="white", palette=None)

        H, L = 1, 1
        fig, ax = plt.subplots(H, L, figsize = (L*7, H*6), sharex=False, sharey=False)
        ax.axis("off")

        N = len(x)
        viridis = plt.get_cmap(cmap)
        colors = linspace(v = viridis.colors, N = N)
        viridis = ListedColormap(colors)
        sns.scatterplot(ax = ax, x = data["x"], y = data["y"], palette=colors, legend = False)
        ax.legend(frameon=False)
        sns.set()

    elif option == 2 :

        # figsize=(2*5,1*5)
        # fig, (ax1, ax2) = plt.subplots(1, 2, sharex=False, sharey=False, figsize = figsize)
        # fig.suptitle("expected vs predicted neighbors")

        # ax1.scatter(x, v1, marker='*', label="expected (i-1)")
        # ax1.scatter(x, v2, marker='o', label="expected (i+1)")
        # ax1.scatter(x, v1_hat, marker='.', label="predicted")
        # ax1.set_xlabel("i")
        # ax1.set_ylabel("neighbors")
        # ax1.legend()
        # ax1.set_title('first neighbor')

        # ax2.scatter(x, v1, marker='*', label="expected (i-1)")
        # ax2.scatter(x, v2, marker='o', label="expected (i+1)")
        # ax2.scatter(x, v2_hat, marker='.', label="predicted")
        # ax2.set_xlabel("i")
        # ax2.set_ylabel("neighbors")
        # ax2.legend()
        # ax2.set_title('2nd neighbor')

        # plt.savefig('/content/expected_vs_predicted_neighbors.png')

        data = {"x" : x, "v1" : v1, "v2" : v2, "v1_hat" : v1_hat, "v2_hat" : v2_hat}
        data = pd.DataFrame.from_dict(data)

        #sns.set(style="whitegrid")
        sns.set_theme(style="white", palette=None)

        H, L = 1, 2
        fig, (ax1, ax2) = plt.subplots(H, L, figsize = (L*7, H*5), sharex=False, sharey=True)
        #fig.suptitle("expected vs predicted neighbors")
        #ax1.axis("off")
        #ax2.axis("off")

        viridis = plt.get_cmap(cmap)
        colors = linspace(v = viridis.colors, N = 50)
        viridis = ListedColormap(colors)

        #sns.scatterplot(ax = ax1, x = data["x"], y = data["v1"], marker='*', label="expected (i-1)", color=tuple(colors[45]), legend = True, s=60)
        #sns.scatterplot(ax = ax1, x = data["x"], y = data["v2"], marker='o', label="expected (i+1)", color=tuple(colors[40]), legend = True, s=40)
        #sns.scatterplot(ax = ax1, x = data["x"], y = data["v1_hat"], marker='.', label="predicted", color=tuple(colors[35]), legend = True, s=80)
        order=1
        sns.regplot(ax=ax1, x=data["x"], y=data["v1"], label="expected (i-1)", order=order,
                    #color=tuple(colors[5]), 
                    marker='*',
                    scatter_kws={'s':10},
                    )
        sns.regplot(ax=ax1, x=data["x"], y=data["v2"], label="expected (i+1)", order=order,
                    #color=tuple(colors[25]), 
                    marker='o',
                    scatter_kws={'s':10}
                    )
        sns.regplot(ax=ax1, x=data["x"], y=data["v1_hat"], label="predicted", order=order,
                    #color=tuple(colors[48]), 
                    marker='.',
                    scatter_kws={'s':10}
                    )

        ax1.set_xlabel("i")
        ax1.set_ylabel("neighbors")
        ax1.legend(frameon=False)
        ax1.set_title('a) first neighbors', y=-0.2)

        # sns.scatterplot(ax = ax2, x = data["x"], y = data["v1"], marker='*', label="expected (i-1)", color=tuple(colors[45]), legend = True, s=60)
        # sns.scatterplot(ax = ax2, x = data["x"], y = data["v2"], marker='o', label="expected (i+1)", color=tuple(colors[40]), legend = True, s=40)
        # sns.scatterplot(ax = ax2, x = data["x"], y = data["v2_hat"], marker='.', label="predicted", color=tuple(colors[35]), legend = True, s=80)

        sns.regplot(ax=ax2, x=data["x"], y=data["v1"], label="expected (i-1)", order=order,
                    #color=tuple(colors[15]), 
                    marker='*',
                    scatter_kws={'s':10}
                    )
        sns.regplot(ax=ax2, x=data["x"], y=data["v2"], label="expected (i+1)", order=order,
                    #color=tuple(colors[25]), 
                    marker='o',
                    scatter_kws={'s':10}
                    )
        sns.regplot(ax=ax2, x=data["x"], y=data["v2_hat"], label="predicted", order=order,
                    #color=tuple(colors[33]), 
                    marker='.',
                    scatter_kws={'s':10}
                    )

        ax2.set_xlabel("i")
        #ax2.set_ylabel("neighbors")
        ax2.set_ylabel("")
        ax2.legend(frameon=False)
        ax2.set_title('b) second neighbors', y=-0.2)

        sns.despine()

        plt.savefig(f'{save_path}/expected_vs_predicted_neighbors.png')
        if SAVE_IN_PGF: plt.savefig(f'{save_path}/expected_vs_predicted_neighbors.pgf')

    else :
        x = torch.arange(1, N-1)

        y1 = torch.zeros_like(x).float().numpy()
        y2 = torch.zeros_like(x).float().numpy()

        y1_hat = torch.zeros_like(x).float().numpy()
        y2_hat = torch.zeros_like(x).float().numpy()

        x = x.numpy()

        for i in range(N-2) :
            tmp = v[i].tolist()
            tmp_hat = v_hat[i].tolist()
            #print(i+1, tmp, tmp_hat)
            E_i = E[i+1]

            d1 = ((E_i - E[tmp[0]])**2).sum().sqrt().item()
            d2 = ((E_i - E[tmp[1]])**2).sum().sqrt().item()

            d1_hat = ((E_i - E[tmp_hat[0]])**2).sum().sqrt().item()
            d2_hat = ((E_i - E[tmp_hat[1]])**2).sum().sqrt().item()

            y1[i] = d1
            y2[i] = d2
            y1_hat[i] = d1_hat
            y2_hat[i] = d2_hat

    if option == 3 :
        data = {"x" : x, "y1" : y1, "y2" : y2, "y1_hat" : y1_hat, "y2_hat" : y2_hat}
        data = pd.DataFrame.from_dict(data)

        #sns.set(style="whitegrid")
        sns.set_theme(style="white", palette=None)

        H, L = 1, 2
        fig, (ax1, ax2) = plt.subplots(H, L, figsize = (L*7, H*5), sharex=False, sharey=True)
        #fig.suptitle("expected vs predicted distances")

        flag = True
        flag2 = True
        flag_sns = True

        order = 1

        viridis = plt.get_cmap(cmap)
        colors = linspace(v = viridis.colors, N = 50)
        viridis = ListedColormap(colors)

        if flag :
            if flag2 :
                if flag_sns :
                    # sns.scatterplot(ax = ax1, x = data["x"], y = data["y1"], marker='*', label="expected (i-1)", 
                    #                 #color=tuple(colors[0]), 
                    #                 legend = True, s=10)
                    # sns.scatterplot(ax = ax1, x = data["x"], y = data["y2"], marker='o', label="expected (i+1)", 
                    #                 #color=tuple(colors[15]), 
                    #                 legend = True, s=10)
                    #sns.scatterplot(ax = ax1, x = data["x"], y = data["y1_hat"], marker='.', label="predicted", color=tuple(colors[30]), legend = True, s=80)

                    sns.regplot(ax=ax1, x=data["x"], y=data["y1"], label="expected (i-1)", order=order,
                        #color=tuple(colors[5]), 
                        marker='*',
                        scatter_kws={'s':10},
                        fit_reg = False
                    )
                    sns.regplot(ax=ax1, x=data["x"], y=data["y2"], label="expected (i+1)", order=order,
                        #color=tuple(colors[5]), 
                        marker='o',
                        scatter_kws={'s':10},
                        fit_reg = False
                    )
                    sns.regplot(ax=ax1, x=data["x"], y=data["y1_hat"], label="predicted", order=order,
                        #color=tuple(colors[5]), 
                        marker='.',
                        scatter_kws={'s':10},
                    )
                else :
                    ax1.scatter(x, y1, marker='*', label="expected (i-1)")
                    ax1.scatter(x, y2, marker='o', label="expected (i+1)")
                    ax1.scatter(x, y1_hat, marker='.', label="predicted")
                ax1.set_xlabel("i")
                ax1.set_ylabel("distances")
            else :
                #ax1.scatter(y1, y1_hat, label="expected") 
                sns.regplot(x=y1, y=y1_hat, color="g", marker='*', ax=ax1, label="predicted vs expected (i-1)", order=order)
                sns.regplot(x=y2, y=y1_hat, color="r", marker='.', ax=ax1, label="predicted vs expected (i+1)", order=order)
                ax1.set_xlabel("predicted") 
                ax1.set_ylabel("expected") 
        else :
            if flag_sns :
                sns.lineplot(ax = ax1, x=data["x"], y=data["y1"], label="expected (i-1)")
                sns.lineplot(ax = ax1, x=data["x"], y=data["y2"], label="expected (i+1)")
                sns.lineplot(ax = ax1, x=data["x"], y=data["y1_hat"], label="predicted")
            else :
                ax1.plot(x, y1, label="expected (i-1)")
                ax1.plot(x, y2, label="expected (i+1)") 
                ax1.plot(x, y1_hat, label="predicted") 
            ax1.set_xlabel("i")
            ax1.set_ylabel("distances")

        ax1.legend(frameon=False)
        ax1.set_title('a) first neighbors', y=-0.2)

        if flag :
            if flag2 :
                if flag_sns :        
                    # sns.scatterplot(ax = ax2, x = data["x"], y = data["y1"], marker='*', 
                    #                 #color=tuple(colors[5]), 
                    #                 label="expected (i-1)", legend = True, s=60)
                    # sns.scatterplot(ax = ax2, x = data["x"], y = data["y2"], marker='o', 
                    #                 #color=tuple(colors[15]), 
                    #                 label="expected (i+1)", legend = True, s=40)
                    # sns.scatterplot(ax = ax2, x = data["x"], y = data["y2_hat"], marker='.', color=tuple(colors[30]), label="predicted", legend = True, s=80)
                
                    sns.regplot(ax=ax2, x=data["x"], y=data["y1"], label="expected (i-1)", order=order,
                        #color=tuple(colors[5]), 
                        marker='*',
                        scatter_kws={'s':10},
                        fit_reg = False
                    )
                    sns.regplot(ax=ax2, x=data["x"], y=data["y2"], label="expected (i+1)", order=order,
                        #color=tuple(colors[5]), 
                        marker='o',
                        scatter_kws={'s':10},
                        fit_reg = False
                    )
                    sns.regplot(ax=ax2, x=data["x"], y=data["y2_hat"], label="predicted", order=order,
                        #color=tuple(colors[5]), 
                        marker='.',
                        scatter_kws={'s':10},
                    )

                else :
                    ax2.scatter(x, y1, marker='*', label="expected (i-1)")
                    ax2.scatter(x, y2, marker='o', label="expected (i+1)")
                    ax2.scatter(x, y2_hat, marker='.', label="predicted")
                ax2.set_xlabel("i")
                #ax2.set_ylabel("distances")
                ax2.set_ylabel("")
            else :
                #ax2.scatter(y2, y2_hat, label="predicted")
                sns.regplot(x=y1, y=y2_hat, color="g", marker='*', ax=ax2, label="predicted vs expected (i-1)", order=order)
                sns.regplot(x=y2, y=y2_hat, color="r", marker='.', ax=ax2, label="predicted vs expected (i+1)", order=order)
                ax2.set_xlabel("predicted")
                ax2.set_ylabel("expected") 
        else :
            if flag_sns : 
                sns.lineplot(ax = ax2, x=data["x"], y=data["y1"], label="expected (i-1)")
                sns.lineplot(ax = ax2, x=data["x"], y=data["y2"], label="expected (i+1)")
                sns.lineplot(ax = ax2, x=data["x"], y=data["y2_hat"], label="predicted")
            else :
                ax2.plot(x, y1, label="expected (i-1)")
                ax2.plot(x, y2, label="expected (i+1)") 
                ax2.plot(x, y2_hat, label="predicted")
            ax2.set_xlabel("i")
            ax2.set_ylabel("distances")

        ax2.legend(frameon=False)
        ax2.set_title('b) second neighbors', y=-0.2)
        sns.despine()

        plt.savefig(f'{save_path}/expected_vs_predicted_distances.png')
        if SAVE_IN_PGF: plt.savefig(f'{save_path}/expected_vs_predicted_distances.pgf')
    
    elif option == 4 :
        IDs = np.zeros_like(y1)
        IDs_hat = np.zeros_like(y1)

        epsilon = 1e-8 

        for i in range(N-2) :
            tmp = v[i].tolist()
            tmp_hat = v_hat[i].tolist()
            #print(i+1, tmp, tmp_hat)
            E_i = E[i+1]

            d1 = ((E_i - E[tmp[0]])**2).sum().sqrt().item()
            d2 = ((E_i - E[tmp[1]])**2).sum().sqrt().item()

            d1_hat = ((E_i - E[tmp_hat[0]])**2).sum().sqrt().item()
            d2_hat = ((E_i - E[tmp_hat[1]])**2).sum().sqrt().item()

            IDs[i] =  max(d1, d2) / (min(d1, d2) + epsilon)
            IDs_hat[i] = max(d1_hat, d2_hat) / (min(d1_hat, d2_hat) + epsilon)

            IDs[i] = np.log(IDs[i])**-1
            IDs_hat[i] = np.log(IDs_hat[i])**-1

        print(IDs.sum(), IDs_hat.sum())

        plt.figure(figsize=(5,5))
        plt.scatter(x, IDs, marker='+')
        plt.scatter(x, IDs_hat, marker='o')
        #plt.scatter(IDs, IDs_hat, marker='o')
        plt.show()