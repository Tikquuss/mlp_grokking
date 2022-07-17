# torch
import torch
import torch.nn as nn

# PyTorch Lightning
import pytorch_lightning as pl

# wandb
import wandb

import itertools
import math

## ID
from intrinsics_dimension import twonn_pytorch, mle_id
ID_functions = {"twonn" : twonn_pytorch, "mle" : mle_id}

possible_metrics = ["%s_%s"%(i, j) for i, j in itertools.product(["train", "val"], ["acc", "loss"])]

from .optim import configure_optimizers

def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m

def make_mlp(l, act=nn.LeakyReLU(), tail=[]):
    """makes an MLP with no top layer activation"""
    return nn.Sequential(*(sum(
        [[nn.Linear(i, o)] + ([act] if n < len(l)-2 else [])
         for n, (i, o) in enumerate(zip(l, l[1:]))], []) + tail))

class Model(pl.LightningModule):
    """
    params : 
        - p (int), emb_dim (int), hidden_dim (int),  n_layers (int), regression (bool), modular (bool)
        - pad_index (int, optional, None), use_wandb (int, optional, False)
        - representation_lr and decoder_lr (float, optional, 1e-3), weight_decay (float, optional, 0) 
        - factor (float, optional, 0.2), patience (float, optional, 20), min_lr (float, optional, 5e-5)
        - E_factor (float, optional, 1.0) : E = (E_a + E_b) / E_factor
    """
    def __init__(self, params):
        """
        Transformer model 
        """
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters(params) 

        self.embeddings = Embedding(self.hparams.p, self.hparams.emb_dim, padding_idx=self.hparams.get("pad_index", None))
        self.embeddings_dropout = nn.Dropout(self.hparams.representation_dropout)
        
        self.operator = self.hparams.operator
        mm = 2*(self.hparams.p-1)+1 if self.operator == "+" else (self.hparams.p-1)**2+1
        mm = self.hparams.p if self.hparams.modular else mm
        mm = 1 if self.hparams.regression else mm
        self.mlp = make_mlp([self.hparams.emb_dim] + [self.hparams.hidden_dim] * self.hparams.n_layers + [mm])
        self.mlp_dropout = nn.Dropout(self.hparams.decoder_dropout)

        self.criterion = nn.MSELoss() if self.hparams.regression else nn.CrossEntropyLoss() 

        ID_params = {**{}, **self.hparams.get("ID_params", {"method" : "mle", "k":2})}
        #ID_params = {"method" : "twonn"}
        id_funct = ID_functions.get(ID_params.pop("method", None), None)
        self.ID_function = id_funct
        setattr(self, "ID", id_funct is not None)
        self.ID_params = ID_params

        self.use_wandb = self.hparams.use_wandb

        self.E_factor = self.hparams.get("E_factor", 1.0)

        # State
        self.grok = False
        self.comprehension = False
        self.memorization = False
        self.confusion = True
        self.comp_epoch = float("inf")
        self.memo_epoch = float("inf")

        # Early stopping grokking : Stop the training `patience` epochs after the `metric` has reached the value `metric_threshold`
        early_stopping_grokking = self.hparams.early_stopping_grokking
        if type(early_stopping_grokking) != dict : early_stopping_grokking = {} 
        self.es_patience = early_stopping_grokking.get("patience", self.hparams.max_epochs)
        self.es_metric = early_stopping_grokking.get("metric", "val_loss" if self.hparams.regression else "val_acc") 
        assert self.es_metric in possible_metrics
        self.es_metric_threshold = early_stopping_grokking.get("metric_threshold", 0.0 if 'loss' in self.es_metric else 99.0) 
        self.es_mode = (lambda s : "min" if 'loss' in s else 'max')(self.es_metric)
        self.es_step = 0
        self.reached_limit = False

    def configure_optimizers(self):
        parameters = [
            {'params': self.embeddings.parameters(), 'lr': self.hparams.get("representation_lr", 1e-3)}, 
            {'params': self.mlp.parameters(), 'lr': self.hparams.get("decoder_lr", 1e-3)}
        ]
        return configure_optimizers(parameters, self.hparams.optimizer, self.hparams.lr_scheduler)

    def forward(self, x):
        """
        Inputs: `x`, LongTensor(bs, 2), containing word indices
        """
        a, b = x[...,0], x[...,1] # (bs,)
        #a, b = a.unsqueeze(1), b.unsqueeze(1) # (bs, 1)
        E_a = self.embeddings_dropout(self.embeddings(a)) # (bs, emb_dim)
        E_b = self.embeddings_dropout(self.embeddings(b)) # (bs, emb_dim)
        E = (E_a + E_b) if self.operator == "+" else E_a*E_b
        E = E / self.E_factor
        tensor = self.mlp_dropout(self.mlp(E)).squeeze() # (bs,) if regression, (bs, 2*(p - 1)+1) if classification
        return tensor, E_a, E_b
    
    def _get_loss(self, batch):
        """
        Given a batch of data, this function returns the  loss (MSE or CEL)
        """
        x, y = batch # We do not need the labels
        tensor, E_a, E_b = self.forward(x)
        loss = self.criterion(input = tensor, target=y)
        return loss, E_a, E_b, tensor, y
    
    def training_step(self, batch, batch_idx):
        loss, E_a, E_b, tensor, y = self._get_loss(batch)  
        self.log('train_loss', loss, prog_bar=True)
        output = {"loss" : loss, "E_a" : E_a, "E_b" : E_b}
        if not self.hparams.regression : 
            acc = (tensor.argmax(dim=-1) == y).float().mean() * 100
            output["train_acc"] = acc
            self.log('train_acc', acc, prog_bar=True)
        return output 
    
    def validation_step(self, batch, batch_idx):
        loss, E_a, E_b, tensor, y = self._get_loss(batch)
        self.log('val_loss', loss, prog_bar=True)
        output = {'val_loss' : loss, "E_a" : E_a, "E_b" : E_b}
        if not self.hparams.regression : 
            acc = (tensor.argmax(dim=-1) == y).float().mean() * 100
            output["val_acc"] = acc
            self.log('val_acc', acc, prog_bar=True)
        return  output 
    
    def test_step(self, batch, batch_idx):
        loss, E_a, E_b, tensor, y = self._get_loss(batch)
        self.log('test_loss', loss, prog_bar=True)
        output = {'test_loss' : loss, "E_a" : E_a, "E_b" : E_b}
        if not self.hparams.regression : 
            acc = (tensor.argmax(dim=-1) == y).float().mean() * 100
            output["test_acc"] = acc
            self.log('test_acc', acc, prog_bar=True)
        return output 

    def _group_E(self, outputs):
        """
        Merges the embedding representation from all batches in one epoch.
        """ 
        E_a = torch.cat([output["E_a"] # (batch_size, embed_dim)
                                 for output in outputs], dim=0) # (n, embed_dim)

        E_b = torch.cat([output["E_b"] # (batch_size, embed_dim)
                                 for output in outputs], dim=0) # (n, embed_dim)

        return E_a, E_b

    @staticmethod
    def compute_intrinsic_dimension(self, outputs, batch_size = 1000):
        """
        Estimate intrinsic dimensions using all bottlenecks collected across one epoch
        bottlenecks : (n, latent_dim)    
        """
        E_a, E_b = self._group_E(outputs)
        
        # to save time, juste use 5000 samples
        E_a, E_b = E_a[:5000], E_b[:5000]

        try :
            return self.ID_function(data=E_a, **self.ID_params), self.ID_function(data=E_b, **self.ID_params)
        except RuntimeError: #CUDA out of memory
            # batchification
            i = 0
            int_dim1, int_dim2 = [], []
            while i < E_a.size(0):
                a = self.ID_function(data=E_a[i: i + batch_size], **self.ID_params)
                b = self.ID_function(data=E_b[i: i + batch_size], **self.ID_params)
                int_dim1.append(a)
                int_dim2.append(b)
            return sum(int_dim1) / len(int_dim1), sum(int_dim2) / len(int_dim2)

    def increase_es_limit(self, logs):
        es_metric = logs[self.es_metric]
        self.reached_limit = self.reached_limit or (es_metric >= self.es_metric_threshold if self.es_mode == "max" 
                                                    else es_metric <= self.es_metric_threshold)
        if self.reached_limit : self.es_step+=1
        return self.es_step

    def training_epoch_end(self, outputs):
        """
        Used by pytorch_lightning
        """
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        logs = {"train_loss": loss}

        if 'train' in self.es_metric : logs["e_step"] = self.increase_es_limit(logs)

        if self.hparams.regression : memo_condition = round(loss.item(), 10) == 0.0
        else : 
            accuracy = torch.stack([x["train_acc"] for x in outputs]).mean()
            logs["train_acc"] = accuracy
            memo_condition = accuracy >= 99.0

        self.memorization = self.memorization or memo_condition
        if memo_condition : self.memo_epoch = min(self.current_epoch, self.memo_epoch)
        
        if self.ID : 
            logs["train_ID_embed"] = self.ID_function(data=self.embeddings.weight, **self.ID_params)
            logs["train_ID_E_a"], logs["train_ID_E_b"] = self.compute_intrinsic_dimension(self, outputs) 
            if not self.hparams.regression :
                logs["train_ID_out_layer"] = self.ID_function(data=self.mlp[-1].weight, **self.ID_params)
            
            #self.mlp[-1].weight # (vocab_len, hidden_dim)
            #self.embeddings.weight # (output_dim, embed_dim)

        logs["train_epoch"]  = self.current_epoch

        schedulers = self.lr_schedulers()
        if schedulers is not None :
            try : scheduler = schedulers[0]
            except TypeError: scheduler = schedulers # 'xxx' object is not subscriptable
            param_groups = scheduler.optimizer.param_groups
            logs["representation_lr"] = param_groups[0]["lr"]
            logs["decoder_lr"] = param_groups[1]["lr"]

        for k, v in logs.items() : self.log(k, v, prog_bar=True)
        if self.use_wandb: wandb.log(logs)

    def validation_epoch_end(self, outputs):
        """
        Used by pytorch_lightning
        """
        loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {
            "val_loss": loss,    
            #"val_epoch": self.current_epoch,
        }

        if self.hparams.regression : comp_condition = round(loss.item(), 10) == 0.0
        else : 
            accuracy = torch.stack([x["val_acc"] for x in outputs]).mean()
            logs["val_acc"] = accuracy
            comp_condition = accuracy >= 99.0
 
        self.comprehension = self.comprehension or comp_condition
        if comp_condition : self.comp_epoch = min(self.current_epoch, self.comp_epoch)

        if 'val' in self.es_metric : logs["e_step"] = self.increase_es_limit(logs)

        if self.ID : 
            logs["val_ID_E_a"], logs["val_ID_E_b"] = self.compute_intrinsic_dimension(self, outputs) 
            # if not self.hparams.regression :
            #     logs["val_ID_out_layer"] = self.ID_function(data=self.mlp[-1].weight, **self.ID_params)
            # logs["val_ID_embed"] = self.ID_function(data=self.embeddings.weight, **self.ID_params)

        for k, v in logs.items() : self.log(k, v, prog_bar=True)
        if self.use_wandb: wandb.log(logs)

        self.grok = self.comprehension and True # and long step of training
        self.memorization = (not self.comprehension) and self.memorization
        self.confusion = (not self.comprehension) and (not self.memorization)

        # diff_epoch = self.comp_epoch - self.memo_epoch
        # if not math.isnan(diff_epoch) : 
        #     self.grok = diff_epoch >= 100
        #     self.comprehension = not self.grok

        self.states = {
            "grok":self.grok, "comprehension":self.comprehension, "memorization": self.memorization, "confusion":self.confusion,
            "comprehension_epoch":self.comp_epoch, "memorization_epoch":self.memo_epoch
        }

    def send_dict_to_wandb(self, data, label, title) :
        if self.hparams.use_wandb:  
            labels = data.keys()
            values = data.values()
            data = [[label, val] for (label, val) in zip(labels, values)]
            table = wandb.Table(data=data, columns = ["label", "value"])
            wandb.log({label : wandb.plot.bar(table, "label", "value", title=title)})
    
    def on_train_start(self):
        db_data = getattr(self.hparams, "data_infos", None)
        if db_data is not None : self.send_dict_to_wandb(db_data, label = "data_info", title="Dataset Informations")

    def on_train_end(self) :

        # diff_epoch = self.comp_epoch - self.memo_epoch
        # if not math.isnan(diff_epoch) : 
        #     self.grok = diff_epoch >= 100
        #     self.comprehension = not self.grok

        states = {
            "grok":int(self.grok), "comprehension":int(self.comprehension), "memorization":int(self.memorization), "confusion":int(self.confusion),
            "comprehension_epoch":self.comp_epoch, "memorization_epoch":self.memo_epoch
        }
        self.send_dict_to_wandb(states, label = "states_info", title="Phase Informations")

if __name__ == "__main__":
    from .utils import AttrDict

    p = 4
    params = AttrDict({
        "p":p, 
        "emb_dim" : 100, 
        "hidden_dim" : 105,  
        "n_layers" : 2,
        "regression" : True,
        "operator" : "+",
        "modular" : True,
        "ID_params" : {},
        "pad_index" : None, 
        "use_wandb" : False
    })

    bs = 4
    x = torch.zeros(size=(bs, 2), dtype=torch.long)
    y = x.sum(1) 
    print(y.shape)

    model = Model(params)
    tensor, E_a, E_b = model(x)
    print(tensor.shape, E_a.shape, E_b.shape, model.criterion(input = tensor, target=y))

    params["regression"] = False
    model = Model(params)
    tensor, E_a, E_b = model(x)
    print(tensor.shape, E_a.shape, E_b.shape, model.criterion(input = tensor, target=y), (tensor.argmax(dim=-1) == y).float().mean())