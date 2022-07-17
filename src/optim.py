# adapter from https://github.com/facebookresearch/XLM/blob/main/xlm/optim.py

import re
import math
import inspect

import torch
from torch import optim

class CustomAdam(optim.Optimizer):
    """
    Same as https://github.com/pytorch/pytorch/blob/master/torch/optim/adam.py,
    without amsgrad, with step in a tensor, and states initialization in __init__.
    It was important to add `.item()` in `state['step'].item()`.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0  # torch.zeros(1)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure=None):
        """
        Step.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # if group['weight_decay'] != 0:
                #     grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                #exp_avg.mul_(beta1).add_(1 - beta1, grad)
                #exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                # Thanks https://github.com/clovaai/AdamP/issues/5
                exp_avg.mul_(beta1).add_(grad, alpha=1-beta1) 
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2) 
                
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                # denom = exp_avg_sq.sqrt().clamp_(min=group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']  # .item()
                bias_correction2 = 1 - beta2 ** state['step']  # .item()
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'] * group['lr'], p.data)

                # TODO : https://github.com/pytorch/pytorch/issues/32861
                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss


class AdamInverseSqrtWithWarmup(CustomAdam):
    """
    Decay the LR based on the inverse square root of the update number.
    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (`warmup-init-lr`) until the configured
    learning rate (`lr`). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.
    During warmup:
        lrs = torch.linspace(warmup_init_lr, lr, warmup_updates)
        lr = lrs[update_num]
    After warmup:
        lr = decay_factor / sqrt(update_num)
    where
        decay_factor = lr * sqrt(warmup_updates)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, warmup_updates=4000, warmup_init_lr=1e-7,
                 exp_factor=0.5):
        super().__init__(
            params,
            lr=warmup_init_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

        # linearly warmup for the first warmup_updates
        self.warmup_updates = warmup_updates
        self.warmup_init_lr = warmup_init_lr
        warmup_end_lr = lr
        self.lr_step = (warmup_end_lr - warmup_init_lr) / warmup_updates

        # then, decay prop. to the inverse square root of the update number
        self.exp_factor = exp_factor
        self.decay_factor = warmup_end_lr * warmup_updates ** self.exp_factor

        # total number of updates
        for param_group in self.param_groups:
            param_group['num_updates'] = 0

    def get_lr_for_step(self, num_updates):
        if num_updates < self.warmup_updates:
            return self.warmup_init_lr + num_updates * self.lr_step
        else:
            return self.decay_factor * (num_updates ** -self.exp_factor)

    def step(self, closure=None):
        super().step(closure)
        for param_group in self.param_groups:
            param_group['num_updates'] += 1
            param_group['lr'] = self.get_lr_for_step(param_group['num_updates'])


class AdamCosineWithWarmup(CustomAdam):
    """
    Assign LR based on a cyclical schedule that follows the cosine function.
    See https://arxiv.org/pdf/1608.03983.pdf for details.
    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (``--warmup-init-lr``) until the configured
    learning rate (``--lr``).
    During warmup::
        lrs = torch.linspace(args.warmup_init_lr, args.lr, args.warmup_updates)
        lr = lrs[update_num]
    After warmup::
        lr = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(t_curr / t_i))
    where ``t_curr`` is current percentage of updates within the current period
    range and ``t_i`` is the current period range, which is scaled by ``t_mul``
    after every iteration.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, warmup_updates=4000, warmup_init_lr=1e-7,
                 min_lr=1e-9, init_period=1000000, period_mult=1, lr_shrink=0.75):
        super().__init__(
            params,
            lr=warmup_init_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

        # linearly warmup for the first warmup_updates
        self.warmup_updates = warmup_updates
        self.warmup_init_lr = warmup_init_lr
        warmup_end_lr = lr
        self.lr_step = (warmup_end_lr - warmup_init_lr) / warmup_updates

        # then, apply cosine scheduler
        self.min_lr = min_lr
        self.max_lr = lr
        self.period = init_period
        self.period_mult = period_mult
        self.lr_shrink = lr_shrink

        # total number of updates
        for param_group in self.param_groups:
            param_group['num_updates'] = 0

    def get_lr_for_step(self, num_updates):
        if num_updates < self.warmup_updates:
            return self.warmup_init_lr + num_updates * self.lr_step
        else:
            t = num_updates - self.warmup_updates
            if self.period_mult == 1:
                pid = math.floor(t / self.period)
                t_i = self.period
                t_curr = t - (self.period * pid)
            else:
                pid = math.floor(math.log(1 - t / self.period * (1 - self.period_mult), self.period_mult))
                t_i = self.period * (self.period_mult ** pid)
                t_curr = t - (1 - self.period_mult ** pid) / (1 - self.period_mult) * self.period
            lr_shrink = self.lr_shrink ** pid
            min_lr = self.min_lr * lr_shrink
            max_lr = self.max_lr * lr_shrink
            return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * t_curr / t_i))

    def step(self, closure=None):
        super().step(closure)
        for param_group in self.param_groups:
            param_group['num_updates'] += 1
            param_group['lr'] = self.get_lr_for_step(param_group['num_updates'])

class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

def get_params_from_string(s : str, separator = ",", have_method=True):
    if "," in s:
        if have_method :
            method = s[:s.find(separator)]
            s = s[s.find(separator) + 1:]
        else :
            method = ""
        optim_params = {}
        for x in s.split(separator):
            split = x.split('=')
            assert len(split) == 2
            try:
                float(split[1])
                assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
                optim_params[split[0]] = float(split[1])
            except ValueError:
                optim_params[split[0]] = split[1]
    else:
        method = s
        optim_params = {}
        
    return method, optim_params

def get_optimizer(parameters, s, noamopt=""):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.001"
        - "adagrad,lr=0.001,lr_decay=0.05"
    noamopt : Optim wrapper that implements rate.
        -"factor_ae=1,warmup_ae=200"
    """
    method, optim_params = get_params_from_string(s, separator = ",", have_method=True)
    #print(method, optim_params)

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method in ['custom_adam', 'adam', 'adam_inverse_sqrt', 'adam_cosine']:
        optim_params['betas'] = (optim_params.get('beta1', 0.9), optim_params.get('beta2', 0.999))
        optim_params.pop('beta1', None)
        optim_params.pop('beta2', None)
        if method == 'custom_adam' :
            optim_fn = CustomAdam 
        elif method == 'adam' :
            optim_fn = optim.Adam
        elif method == 'adam_inverse_sqrt':
            optim_fn = AdamInverseSqrtWithWarmup
        elif method == 'adam_cosine':
            optim_fn = AdamCosineWithWarmup
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    try : 
        expected_args = inspect.getargspec(optim_fn.__init__)[0]
    except ValueError: #Function has keyword-only parameters or annotations, use getfullargspec() API which can support them
        expected_args = inspect.getfullargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    if noamopt != "" :
        _, params = get_params_from_string(s=noamopt, have_method=False)
        return NoamOpt(params["d_model"], params["factor_ae"], params["warmup_ae"], optim_fn(parameters, **optim_params))
    else :
        return optim_fn(parameters, **optim_params)


def get_lr_scheduler(optimizer, s) :
    """
    Parse lr_scheduler parameters.
    Input should be of the form:
        - "reduce_lr_on_plateau,factor=0.2,patience=20,min_lr=0.00005,mode=min,monitor=val_loss"

    """
    method, scheduler_params = get_params_from_string(s, separator = ",", have_method=True)
    monitor = scheduler_params.pop("monitor", "val_loss")
    if method == "reduce_lr_on_plateau" :
        # Reduce learning rate when a metric has stopped improving.
        scheduler_fn =  optim.lr_scheduler.ReduceLROnPlateau
    elif method == "constant_lr" :
        # Decays the learning rate of each parameter group by a small constant factor until the number of epoch reaches a pre-defined milestone: total_iters.
        scheduler_fn =  optim.lr_scheduler.ConstantLR
    elif method == "linear_lr" :
        # Decays the learning rate of each parameter group by linearly changing small multiplicative factor until the number of epoch reaches a pre-defined milestone: total_iters.
        scheduler_fn =  optim.lr_scheduler.LinearLR
    elif method == "cosine_annealing_lr" :
        # Set the learning rate of each parameter group using a cosine annealing schedule, where η_max is set to the initial lr and T_{cur} is the number of epochs since the last restart in SGDR
        scheduler_fn =  optim.lr_scheduler.CosineAnnealingLR
    elif method == "exponential_lr" :
        # Decays the learning rate of each parameter group by gamma every epoch.
        scheduler_fn =  optim.lr_scheduler.ExponentialLR
    elif method == "lambda_lr" :
        # Sets the learning rate of each parameter group to the initial lr times a given function.
        scheduler_fn = optim.lr_scheduler.LambdaLR
    elif method == "multiplicative_lr" :
        # Multiply the learning rate of each parameter group by the factor given in the specified function.
        scheduler_fn = optim.lr_scheduler.MultiplicativeLR
    elif method == "step_lr" :
        # Decays the learning rate of each parameter group by gamma every step_size epochs.
        scheduler_fn = optim.lr_scheduler.StepLR
    elif method == "multi_step_lr" :
        # Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones.
        scheduler_fn = optim.lr_scheduler.MultiStepLR
    elif method == "cyclic_lr" :
        # Sets the learning rate of each parameter group according to cyclical learning rate policy (CLR).
        scheduler_fn = optim.lr_scheduler.CyclicLR
    elif method == "one_cycle_lr" :
        # Sets the learning rate of each parameter group according to the 1cycle learning rate policy.
        scheduler_fn = optim.lr_scheduler.OneCycleLR
    elif method == "cosine_annealing_warm_restarts" :
        # Set the learning rate of each parameter group using a cosine annealing schedule, where η_max is set to the initial lr, T_cur is the number of epochs since the last restart and 
        # T_i is the number of epochs between two warm restarts in SGDR
        scheduler_fn = optim.lr_scheduler.CosineAnnealingWarmRestarts
    # elif method == "chained_scheduler" :
    #     # Chains list of learning rate schedulers.
    #     scheduler_fn = optim.lr_scheduler.ChainedScheduler
    # elif method == "sequential_lr" :
    #     # Receives the list of schedulers that is expected to be called sequentially during optimization process and milestone points that provides exact intervals to reflect which scheduler is supposed to be called at a given epoch.
    #     scheduler_fn = optim.lr_scheduler.SequentialLR
    else:
        raise Exception('Unknown lr scheduler method: "%s"' % method)

    # check that we give good parameters to the optimizer
    try : 
        expected_args = inspect.getargspec(scheduler_fn.__init__)[0]
    except ValueError: #Function has keyword-only parameters or annotations, use getfullargspec() API which can support them
        expected_args = inspect.getfullargspec(scheduler_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'optimizer']
    if not all(k in expected_args[2:] for k in scheduler_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(scheduler_params.keys())))

    return scheduler_fn(optimizer, **scheduler_params), monitor

def configure_optimizers(parameters, s_optim, s_lr_scheduler):
    optimizer = get_optimizer(parameters, s_optim)
    if s_lr_scheduler :
        scheduler, monitor = get_lr_scheduler(optimizer, s_lr_scheduler)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": monitor}
    else :
        return {"optimizer": optimizer}