import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 

def visualize_phases(img, phases, interpolation='nearest', 
                        #figsize=(5,5),
                        title = None, save_to = None, show = True, pixel_wise_text = False
                    ) :
    figsize=(5,5)
    sns.set_theme(style="white", palette=None)
    
    elements, indices, counts = img.unique(return_inverse=True, return_counts=True)
    
    #new_viridis = matplotlib.cm.get_cmap('viridis', elements.size(0)+1)
    new_viridis = matplotlib.cm.get_cmap('viridis', img.max()+1)

    #plt.figure(figsize=(10,10))
    plt.figure(figsize=figsize)
        
    #interpolation='hermite'
    #interpolation=None

    # H, L = 1, 1
    # figsize = tuple([i*j for i, j in zip(figsize, [L, H])])
    # fig, ax = plt.subplots(H, L, figsize = figsize, sharex=False, sharey=False)
    # axi = ax.imshow(img.numpy(), interpolation=interpolation, cmap=new_viridis)
    # cbar=fig.colorbar(mappable=axi, ax=ax)
    # #cbar.minorticks_on()

    axi = plt.imshow(img.numpy(), interpolation=interpolation, cmap=new_viridis)
    img_ratio = img.shape[0]/img.shape[1]
    plt.colorbar(axi, fraction=0.046*img_ratio, pad=0.04)

    inv_phases = {str(v) : str(k) for k, v in phases.items()}
    if pixel_wise_text :
        H, C = img.shape
        xy = torch.from_numpy(np.array(np.meshgrid(np.arange(H), np.arange(C))).T.reshape(-1,2)) # (H*C,2)
        for ij, k in zip(xy, indices.reshape(-1)) :
            i, j = ij
            kkk = elements[k.item()].item()
            plt.text(j,i,f"{kkk}")
            #plt.text(j,i,f"{inv_phases[str(kkk)]}({kkk})")
    else :
        for m, k in  enumerate(elements) :
            #i, j = (img==k.item()).nonzero(as_tuple=False)[0]
            # center the values
            if False :
                tmp1, tmp2 = (img==k.item()).nonzero(as_tuple=True)
                i = torch.div(tmp1.sum(), tmp1.size(0), rounding_mode='trunc')
                j = torch.div(tmp2.sum(), tmp2.size(0), rounding_mode='trunc')
            else :
                tmp = (img==k.item()).nonzero(as_tuple=False) # (H*C, 2)
                mean = tmp.float().mean(dim=0) # (,2)
                ij = ((tmp - mean)**2).sum(dim=1).sqrt().argmin() # ()
                i, j = tmp[ij]
            kkk = k.item()
            plt.text(j.item(),i.item(),f"{kkk}")
            #plt.text(j.item(),i.item(),f"{inv_phases[str(kkk)]}({kkk})")

    if False :
        L, H = figsize
        x1 = -0.3
        x2 = 2*L-7.8 # 2*L-4.5
        y1 = H # 2*H+0.5
        y2 = y1+0.5
        i = 0
        for x in [x1, x2] :
            for y in [y1, y2] :
                plt.text(x, y, f'{i} = {inv_phases[str(i)]}',
                        horizontalalignment='left',
                        verticalalignment='top')
                i+=1
    else :
        title_ = ", ".join([f'{i}={inv_phases[str(i)]}' for i in inv_phases])
        if title is None : title = title_
        else : title = f"{title} ({title_})"

    #plt.legend()
    if title is not None : plt.title(title, y=-0.15)
    if save_to is not None : plt.savefig(save_to)
    sns.set()

    if show : plt.show()
    else : plt.close()

    return img

if __name__ == "__main__":
    import torch

    #decoder_lrs = representation_lrs = [1e-2, 1e-3] 
    decoder_lrs = representation_lrs = np.linspace(start=1e-1, stop=1e-5, num=10)

    weight_decays = list(range(20))
    #weight_decays =  np.linspace(start=0, stop=20, num=21)

    flag = True # if True, decoder_lrs if True, else weight_decays
    if flag : s = "decoder_lr"
    else : s = "weight_decay"

    tmps = decoder_lrs if flag else weight_decays

    M = len(representation_lrs)
    N = len(tmps)
    img = torch.zeros(size=(M,N), dtype=int)

    for i in range(M) :
        for j in range(N) :
            img[i][j] = np.random.randint(low=0, high=3+1)

    phases = {"grokking" : 0, "comprehension" : 1, "memorization" : 2, "confusion" : 3}
    
    _ = visualize_phases(img, phases, interpolation=None, 
                            title = None, save_to = None, show = True, pixel_wise_text = False
    )