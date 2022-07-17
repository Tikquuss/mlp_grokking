import itertools
import numpy as np
import matplotlib.pyplot as plt
import wandb

from src.utils import AttrDict, GROUP_VARS
from src.dataset import get_dataloader
from src.utils import get_group_name
from src.trainer import train

def plot_results(params, model_dict, hparms_1, hparms_2, s1, s2):
    """
    2D plot of train&val acc&loss as a function of two parameters use for phase diagram
    """
    fig = plt.figure()
    fig.suptitle("Grokking")

    figsize=(2*8, 6)
    plt.gcf().set_size_inches(figsize)

    i = 1
    for metric in (["loss"] if params.regression else ["loss", "acc"])  :
        ax = fig.add_subplot(1, 2, i, projection='3d')
        i += 1 
        xs, ys, zs = [], [], []
        for split, (m, zlow, zhigh) in zip(["val", "train"], [('o', -50, -25), ('^', -30, -5)]) :
            for a, b in itertools.product(hparms_1, hparms_2) :
                k = f"{s1}={a}, {s2}={b}"
                if k in model_dict.keys():
                    xs.append(a)
                    ys.append(b)
                    #print(k, f"{split}_{metric}", model_dict[k]["result"][split][f"{split}_{metric}"])
                    zs.append(model_dict[k]["result"][split][f"{split}_{metric}"])

            ax.scatter(xs, ys, zs, marker=m, label = split)

        ax.set_xlabel(s1)
        ax.set_ylabel(s2)
        ax.set_zlabel(metric)
        ax.set_title(metric, fontsize=14)
        ax.legend()
    plt.show()

if __name__ == "__main__":
    
    train_pct=80
    weight_decay=0.0
    representation_lr=0.001
    decoder_lr=0.001
    representation_dropout=0.0
    decoder_dropout=0.0
    opt="adam"

    group_name=f"tdf={train_pct}-wd={weight_decay}-r_lr={representation_lr}-d_lr={decoder_lr}-r_d={representation_dropout}-d_d={decoder_dropout}-opt={opt}"

    random_seed=0
    operator="+"
    modular=False

    log_dir="../log_files"

    p = 100
    task = "classification"

    params = AttrDict({
        ### Main parameters
        "task" : task,
        "exp_id" : f"{task}_{group_name}",
        "log_dir" : f"{log_dir}/{random_seed}",

        ### Model
        "emb_dim" : 256, 
        "hidden_dim" : 512,  
        "n_layers" : 1,
        "representation_dropout" : representation_dropout,
        "decoder_dropout" : decoder_dropout,
        "pad_index" : None, 
        "p" : p, 

        ### Dataset
        "operator" : operator, 
        "modular" : modular,
        "train_pct" : train_pct,
        "batch_size" : 512,

        ### Optimizer
        "optimizer" : f"{opt},weight_decay={weight_decay},beta1=0.9,beta2=0.99,eps=0.00000001",
        "representation_lr" : representation_lr,
        "decoder_lr" : decoder_lr,

        ### LR Scheduler
        "lr_scheduler" : None,
        #"lr_scheduler" : "reduce_lr_on_plateau,factor=0.2,patience=20,min_lr=0.00005,mode=min,monitor=val_loss",
        
        ### Training
        "max_epochs" : 10000, 
        "validation_metrics" : "val_loss",
        "checkpoint_path" : None, 
        "model_name": "None", 
        "every_n_epochs":100, 
        "every_n_epochs_show":200, 
        "early_stopping_patience":1e9, 
        "save_top_k":-1,

        # Wandb 
        "use_wandb" : False,
        "wandb_entity" : "grokking_ppsp",
        "wandb_project" : f"toy_model_grokking_op={operator}-p={p}-task={task}-mod={modular}",
        "group_name" : group_name,

        "group_vars" : None,

        ### Intrinsic Dimension Estimation
        #"ID_params" : {},
        #"ID_params": {"method" : "mle", "k":2},
        "ID_params": {"method" : "twonn"},
        
        ### Devices & Seed
        "accelerator" : "auto",
        "devices" : "auto",
        "random_seed": random_seed,

        ### Early_stopping (for grokking) : Stop the training `patience` epochs after the `metric` has reached the value `metric_threshold` 
        #"early_stopping_grokking" : None,
        "early_stopping_grokking" : "patience=int(1000),metric=str(val_acc),metric_threshold=float(90.0)"
    })
    params["weight_decay"] = weight_decay
    params["regression"] = task == "regression"
    train_loader, val_loader, dataloader, data_infos = get_dataloader(
        p, train_pct, regression = params.regression, operator=params.operator, 
        modular = params.modular, batch_size=params.batch_size, num_workers=2
    )
    print(data_infos, "\n\n")
    params["data_infos"] = data_infos
    ######## Example : phase diagram with representation_lr and decoder_lr/weight_decay

    #decoder_lrs = representation_lrs = [1e-2, 1e-3, 1e-4, 1e-5] 
    decoder_lrs = representation_lrs = np.linspace(start=1e-1, stop=1e-5, num=10)

    #weight_decays = list(range(20))
    weight_decays =  np.linspace(start=0, stop=20, num=21)

    flag = True # if True, decoder_lrs if True, else weight_decays
    if flag : s = "decoder_lr"
    else : s = "weight_decay"
    print(representation_lrs, decoder_lrs if flag else weight_decays)

    model_dict = {}
    i = 0
    for a, b in itertools.product(representation_lrs, decoder_lrs if flag else weight_decays) :

        params["representation_lr"] = a 
        if flag : params[s] = b
        else : params["optimizer"] = params["optimizer"].replace(f"{s}={weight_decay}", f"{s}={b}")
    
        name = f"representation_lr={a}, {s}={b}"
        params.exp_id = name
        
        #group_vars = GROUP_VARS + ["representation_lr", s]
        group_vars = ["representation_lr", s]
        group_vars = list(set(group_vars))
        params["group_name"] = get_group_name(params, group_vars = None)
        
        print("*"*10, i, name, "*"*10)
        i+=1

        model, result = train(params, train_loader, val_loader)
        
        model_dict[name] = {"model": model, "result": result}

    ########

    print(model_dict.keys())

    val_loss = [model_dict[k]["result"]["val"]["val_loss"] for k in model_dict]
    val_acc = [model_dict[k]["result"]["val"].get("val_acc", 0) for k in model_dict]
    print(val_loss, val_acc)

    plot_results(params, model_dict, 
        hparms_1 = representation_lrs, hparms_2 = decoder_lrs if flag else weight_decays,
        s1 = 'representation_lr', s2 = s
    )

    ########

    # for k in model_dict :
    #     print("*"*10, k, "*"*10)
    #     model = model_dict[k]["model"]
    #     # TODO