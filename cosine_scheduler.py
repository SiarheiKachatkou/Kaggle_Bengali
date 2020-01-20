import math
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmUpRestarts():
    def __init__(self,period,lr_initial,warm_up_period):

        self.period, self.lr_initial, self.warm_up_period=period,lr_initial,warm_up_period

    def _linear_part(self,time,final_time,final_lr):
        return time*final_lr/final_time

    def _cosine_part(self,time,period,lr_initial):
        return lr_initial*(math.cos(time*2*math.pi/(period*2))+1)/2


    def calc_lr(self,time):
        if time<self.warm_up_period:
            return self._linear_part(time,self.warm_up_period,self.lr_initial)
        else:
            return self._cosine_part(time-self.warm_up_period,self.period-self.warm_up_period,self.lr_initial)

    def is_finished(self,time):
        return time>=self.period


class CosineScheduler(_LRScheduler):
    def __init__(self, optimizer, period_initial, period_mult=1, lr_initial=0.1, period_warmup_percent=0, lr_reduction=1.):
        self.period = period_initial
        self.period_mult = period_mult
        self.lr = lr_initial
        self.lr_initial = lr_initial
        self.period_warmup_percent = period_warmup_percent
        self.lr_reduction = lr_reduction
        self.time_curr = 0
        self.lr_calculator=CosineAnnealingWarmUpRestarts(self.period,self.lr_initial,self.period_warmup_percent*self.period)

        super(CosineScheduler, self).__init__(optimizer, last_epoch = -1)

    def get_lr(self):
        return [self.lr]

    def step(self, epoch=None):
        if epoch is not None:
            raise NotImplementedError

        self.time_curr+=1
        time=self.time_curr
        if self.lr_calculator.is_finished(time):
            self.time_curr=0
            self.period*=self.period_mult
            self.lr_initial*=self.lr_reduction
            self.lr_calculator=CosineAnnealingWarmUpRestarts(self.period,self.lr_initial,self.period_warmup_percent*self.period)

        self.lr=self.lr_calculator.calc_lr(self.time_curr)

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
