# torch & torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from copy import copy

import torchvision
import torchvision.transforms.functional as F

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

import wandb

import cv2
import os
import re 

from .visualize_with_plotly import display_pca_scatterplot, figure_to_images
from .visualize import visualize_embeddings_good
from .modeling import Model
from .utils import get_group_name, init_wandb


# Thanks https://pytorch.org/vision/stable/auto_examples/plot_visualization_utils.html#sphx-glr-auto-examples-plot-visualization-utils-py
plt.rcParams["savefig.bbox"] = 'tight'
def do_show(imgs, figsize=(14,14), show = True, title = None, save_to = None):
    sns.set_theme(style="white", palette=None)
    if not isinstance(imgs, list): imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), figsize=figsize, squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if title is not None : plt.title(title, y=-0.13)
    if save_to is not None : plt.savefig(save_to)
    sns.set()

    if show : plt.show()
    else : plt.close()


class GenerateCallback(pl.Callback):
    """Use to plot the learned input embeddings at different training stages"""
    
    def __init__(self, log_dir, dim=2, pca=True, every_n_epochs=1, every_n_epochs_show=10, format="png"):
        super().__init__()
        self.dim = dim
        self.pca = pca
        self.every_n_epochs = every_n_epochs # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.every_n_epochs_show = every_n_epochs_show
        self.log_dir = os.path.join(log_dir, "images")
        self.log_dir_E = os.path.join(self.log_dir, "E")
        self.log_dir_embed = os.path.join(self.log_dir, "embeddings")
        self.log_dir_layer_layer_weight = os.path.join(self.log_dir, "layer_layer_weight")
        self.learned_set_of_embeddings = os.path.join(self.log_dir, "learned_set_of_embeddings")

        for dir in [self.log_dir, self.log_dir_E, self.log_dir_embed, self.log_dir_layer_layer_weight,  self.learned_set_of_embeddings] : 
            os.makedirs(dir, exist_ok = True)
        
        self.format = format

        self.show = False
        #self.show = not pl_module.use_wandb
        
    def on_epoch_end(self, trainer, pl_module):
    #def on_train_epoch_end(self, trainer, pl_module) :
    #def on_validation_epoch_end(self, trainer, pl_module) :
        current_epoch = trainer.current_epoch
        if current_epoch % self.every_n_epochs == 0 :
            plot_figure = display_pca_scatterplot(pl_module.hparams.p, pl_module, dim=self.dim, pca=self.pca, perplexity=5, learning_rate=500, 
                                                  iteration=10000, title = f"E, epoch={current_epoch}", show = self.show)
            file = os.path.join(self.log_dir_E, f"{current_epoch}.{self.format}")
            image1 = figure_to_images(plot_figure, file=file, format=self.format, width=None, height=None, scale=None, validate=True)
            #trainer.logger.experiment.add_image("E", image1, global_step=trainer.global_step)
            
            word_vectors = pl_module.embeddings.weight
            plot_figure = display_pca_scatterplot(pl_module.hparams.p, pl_module, word_vectors = word_vectors, dim=2, pca=True, 
                                                  title=f"embeddings, epoch={current_epoch}", show = self.show)
            file = os.path.join(self.log_dir_embed, f"{current_epoch}.{self.format}")
            image2 = figure_to_images(plot_figure, file=file, format=self.format, width=None, height=None, scale=None, validate=True)
            #trainer.logger.experiment.add_image("embeddings", image2, global_step=trainer.global_step)

            H, W, _ = image1.shape
            images_ = [image1, image2] # H x W x C
            nrow=2
            if not pl_module.hparams.regression :
                word_vectors = pl_module.mlp[-1].weight
                plot_figure = display_pca_scatterplot(word_vectors.size(0), pl_module, word_vectors = word_vectors, dim=2, pca=True, 
                                                      title = f"last_layer_weight, epoch={current_epoch}", show = self.show)
                file = os.path.join(self.log_dir_layer_layer_weight, f"{current_epoch}.{self.format}")
                image3 = figure_to_images(plot_figure, file=file, format=self.format, width=None, height=None, scale=None, validate=True)
                #trainer.logger.experiment.add_image("layer_layer_weight", image2, global_step=trainer.global_step)
                images_.append(image3)
                nrow += 1

            if True :
                images = [img.transpose(2, 0, 1) for img in images_] # C x H x W
                imgs = [torch.from_numpy(img) for img in images] # B x C x H x W, B = 1
            else :
                imgs = [torch.from_numpy(img).transpose(2, 0).transpose(1, 2) for img in images_] # B x C x H x W, B = 1

            if pl_module.hparams.emb_dim == 2 and not pl_module.hparams.regression :
                save_to = os.path.join(self.learned_set_of_embeddings, "LSE.png")
                img = visualize_embeddings_good(pl_module, A = None, B = None, N = 500,
                                                interpolation=None,
                                                #interpolation='hermite',
                                                figsize=(5,5), 
                                                title = f"learned_set_of_embeddings, epoch={current_epoch}",
                                                save_to = save_to,
                                                show = False
                                        ) 
                
                #img = mpimg.imread(save_to)
                #images_.append(img)

                #img = torch.from_numpy(img).transpose(2, 0).transpose(1, 2) 
                img = torchvision.io.read_image(save_to)
                img = F.resize(img, size = (H, W))
                imgs.append(img)
                #nrow+=1
                nrow=4

                img = img.transpose(2, 1).transpose(0, 2).numpy()
                images_.append(img)

            grid = torchvision.utils.make_grid(imgs, nrow=nrow)
            trainer.logger.experiment.add_image("representation_projection", grid, global_step=trainer.global_step)

            do_show(grid, 
                    #show = True, 
                    show = current_epoch % self.every_n_epochs_show == 0,
                    title = f"epoch={current_epoch}", 
                    save_to =  os.path.join(self.log_dir, f"grid_{current_epoch}.png")
            )

            if pl_module.use_wandb : wandb.log({"embedding_projection":  [wandb.Image(img) for img in images_]})


def sorted_nicely(l): 
    """ Sort the given iterable in the way that humans expect.
    https://stackoverflow.com/a/2669120/11814682
    """ 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)
    
def images_to_vidoe(image_folder, video_path, format="png") :
    """Thanks https://stackoverflow.com/a/44948030/11814682"""
    images = [img for img in sorted_nicely(os.listdir(image_folder))  if img.endswith(f".{format}")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_path, 0, 1, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


def train(params, train_loader, val_loader):

    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(params.log_dir, params.exp_id) 
    pp = vars(params)
    trainer_config = {
        "max_epochs": params.max_epochs,
        "default_root_dir" : root_dir,

        "limit_train_batches" : pp.get("limit_train_batches", 1.0), 
        "limit_val_batches" : pp.get("limit_val_batches", 1.0),
        "limit_test_batches": pp.get("limit_test_batches", 1.0),

        "accelerator" : params.accelerator,
        "devices" : params.devices,
        #"reload_dataloaders_every_n_epochs" : True,
        "weights_summary":"full", # "top", None,

        # "log_every_n_steps" : max(len(train_loader) // params.batch_size, 0),
        # "weights_save_path" : os.path.join(root_dir, "weights"),
        # "auto_scale_batch_size" : True, # None
        # "auto_select_gpus" : True,
        # "auto_lr_find": True,
        # "benchmark" : False,
        # "deterministic" : True,
        # "val_check_interval" : 1.,
        # "accumulate_grad_batches" : False,
        # "strategy": "ddp", # "ddp_spaw"
    }

    validation_metrics = params.validation_metrics
    mode = (lambda s : "min" if 'loss' in s else 'max')(validation_metrics)
    early_stopping_callback = EarlyStopping(
        monitor=validation_metrics, patience=params.early_stopping_patience, verbose=False, strict=True,
        mode = mode
    )

    model_checkpoint_callback = ModelCheckpoint(
            dirpath=root_dir,
            save_weights_only=True,
            filename="{epoch}-{%s:.4f}"%validation_metrics,
            mode = mode,
            monitor=validation_metrics,
            save_top_k=params.save_top_k,
    )

    trainer_config["callbacks"] = [
        early_stopping_callback, 
        model_checkpoint_callback,
        GenerateCallback(
            log_dir = root_dir,
            every_n_epochs = params.every_n_epochs,
            every_n_epochs_show = params.every_n_epochs_show
        ), 
        LearningRateMonitor("epoch")
    ]

    trainer = pl.Trainer(**trainer_config)
    
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need
    
    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = root_dir + params.model_name
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model %s, loading..."%pretrained_filename)
        model = Model.load_from_checkpoint(pretrained_filename)
        print(model)
    else:
        # Initialize wandb
        if params.group_name is None : params.group_name = get_group_name(params, group_vars = params.group_vars)
        init_wandb(params.use_wandb, wandb_project = params.wandb_project, group_name = params.group_name, wandb_entity = params.wandb_entity)

        model = Model(params)
        print(model)
        trainer.fit(model, train_loader, val_loader, ckpt_path=params.checkpoint_path)
        model = Model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

        #try : wandb.finish(exit_code = None, quiet = None)
        #except : pass

    # Test best model on validation set
    val_result = trainer.test(model, val_loader, verbose=False)
    train_result = trainer.test(model, train_loader, verbose=False)

    result = {"train": train_result, "val": val_result}
    for k1, v1 in copy(result).items() :
        #for k2 in v1[0] : result[k1][0][k2.replace("test", k1)] = round(result[k1][0].pop(k2), 4)
        result[k1] = {k2.replace("test", k1): round(result[k1][0][k2], 4) for k2 in v1[0]}
    
    return model, result