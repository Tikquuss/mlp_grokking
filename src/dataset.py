import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset

def get_dataloader(p, train_pct, regression, operator="+", ij_equal_ji = True, modular = True, batch_size=256, num_workers=4):
    """We define a data constructor that we can use for various purposes later."""
    assert operator in ["+", "*"]
    if ij_equal_ji :
        x = []
        for i in range(p) :
          for j in range(i, p) :
              x.append([i, j])
        x = torch.LongTensor(x) # (p*(p+1)/2, 2)
    else :
        ij = torch.arange(p) # (p,)
        x = torch.cartesian_prod(ij, ij) # (p^2, 2)
    y = x.sum(1) if operator=="+" else x.prod(1) # (p*(p+1)/2,) if ij_equal_ji, else # (p^2,)
    if modular : y = torch.remainder(y, p)
    if regression : y = y.float() 
  
    dataset = TensorDataset(x, y)

    n = len(dataset)
    train_size = train_pct * n // 100
    val_size = n - train_size

    print(f"train_size, val_size : {train_size}, {val_size}")

    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=min(batch_size, train_size), shuffle=True, drop_last=False, pin_memory=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=min(batch_size, val_size), shuffle=False, drop_last=False, num_workers=num_workers)

    dataloader = DataLoader(dataset, batch_size=min(batch_size, n), shuffle=False, drop_last=False, num_workers=num_workers)

    data_infos = {
        "train_batch_size" : min(batch_size, train_size), "val_batch_size" : min(batch_size, val_size), 
        "train_size":train_size, "val_size":val_size, 
        "train_n_batchs":len(train_loader), "val_n_batchs":len(val_loader)
    }

    return train_loader, val_loader, dataloader, data_infos

if __name__ == "__main__":
    p = 4
    train_pct = 80
    train_loader, val_loader, dataloader, data_infos = get_dataloader(p, train_pct,  regression = True, operator="+", ij_equal_ji = True, modular = True, batch_size=256, num_workers=2)

    print(data_infos)
    
    x, y = next(iter(dataloader))
    print(x, y)