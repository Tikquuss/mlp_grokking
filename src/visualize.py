import torch

#%matplotlib inline

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 

def visualize_embeddings_good(model, A = None, B = None, N = 10, interpolation='hermite', figsize=(5,5),
                              title = None, save_to = None, show = True, pixel_wise_text = False
                              ) :
    if model.hparams.regression : return
    sns.set_theme(style="white", palette=None)
    E = model.embeddings.weight.detach().cpu()    
    if A is None : A = E.min(dim=0).values 
    if B is None : B = E.max(dim=0).values
    x, y = np.linspace(start = A[0], stop = B[0], num=N), np.linspace(start = A[1], stop = B[1], num=N)
    xy = np.array(np.meshgrid(x, y)).T.reshape(-1,2) # (N*N,2)
    # x, y = xy[:,0], xy[:,1]
    xy = torch.from_numpy(xy).float() / model.E_factor
    x, y = xy.unbind(dim=1)
    tensor = model.mlp(xy).squeeze()
    if not model.hparams.regression : z = tensor.softmax(dim=1).argmax(dim=1)
    else : z = tensor.detach()#.round()
    img = z.reshape(N, N).cpu().numpy()

    img = torch.from_numpy(img)
    elements, indices, counts = img.unique(return_inverse=True, return_counts=True)

    #new_viridis = matplotlib.cm.get_cmap('viridis', elements.size(0)+1)
    new_viridis = matplotlib.cm.get_cmap('viridis', img.max()+1)

    #plt.figure(figsize=(10,10))
    plt.figure(figsize=figsize)
    
    #interpolation='hermite'
    #interpolation=None
    axi = plt.imshow(img.numpy(), interpolation=interpolation, cmap=new_viridis)
    plt.colorbar(axi)

    if pixel_wise_text :
        H, C = img.shape
        xy = torch.from_numpy(np.array(np.meshgrid(np.arange(H), np.arange(C))).T.reshape(-1,2)) # (H*C,2)
        for ij, k in zip(xy, indices.reshape(-1)) :
            i, j = ij
            plt.text(j,i,f"{elements[k.item()].item()}")
            #plt.text(i,j,f"{elements[k.item()].item()}({counts[k].item()})")
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

            plt.text(j.item(),i.item(),f"{k.item()}")
            #plt.text(j.item(),i.item(),f"{k.item()}({counts[m].item()})") 

    #plt.legend()
    if title is not None : plt.title(title, y=-0.15)
    if save_to is not None : plt.savefig(save_to)
    sns.set()

    if show : plt.show()
    else : plt.close()

    return img


if __name__ == "__main__":
    from .utils import AttrDict
    from .dataset import get_dataloader
    from .modeling import Model

    p =  15
    params = AttrDict({
        "p":p, 
        "emb_dim" : 2, 
        "hidden_dim" : 105,  
        "n_layers" : 1,
        "regression" : False,
        "operator" : "+",
        "modular" : True,
        "ID_params" : {},
        "pad_index" : None, 
        "use_wandb" : False,
        "E_factor" : 1
    })

    train_pct = 80
    train_loader, val_loader, dataloader, _ = get_dataloader(p, train_pct, regression = params.regression, operator=params.operator, 
                                                        modular = params.modular, batch_size=256, num_workers=2)

    model = Model(params)


    save_to='../d1.png'
    img = visualize_embeddings_good(model, A = None, B = None, N = 10, 
                                    interpolation=None,
                                    #interpolation='hermite',
                                    figsize=(5,5), title = "learned_set_of_embeddings",
                                    save_to=save_to,
                                    pixel_wise_text = True
                                    ) 

    save_to='../d2.png'
    img = visualize_embeddings_good(model, A = None, B = None, N = 100, 
                                    interpolation=None,
                                    #interpolation='hermite',
                                    figsize=(5,5), title = "learned_set_of_embeddings",
                                    save_to=save_to,
                                    pixel_wise_text = False
                                    ) 

    import matplotlib.image as mpimg
    img = mpimg.imread(save_to)
    plt.imshow(img), plt.show()