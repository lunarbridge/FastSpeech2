import torch
import torch.nn as nn
from fastspeech2.trainers.base_trainer import BaseTrainer
from fastspeech2.utils.tools import log
from fastspeech2.utils.tools import to_device


class Trainer(BaseTrainer):
    def __init__(self, model: nn.Module,
                optimizer, train_dataset, valid_dataset,
                max_step: int, save_interval: int, log_interval: int,
                pitch_feature: str, energy_feature: str,
                save_dir: str, save_prefix: str = '',
                grad_clip: float = 0.0, grad_norm: float = 0.0,
                sr: int = 22050, pretrained_path: str = None, scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                seed: int = 2021, is_reference: bool = False):
        super().__init__(model, optimizer, train_dataset, valid_dataset,
                         max_step=max_step, save_interval=save_interval,
                         log_interval=log_interval, save_dir=save_dir, 
                         save_prefix=save_prefix, grad_clip=grad_clip,
                         grad_norm=grad_norm, pretrained_path=pretrained_path,
                         pitch_feature=pitch_feature, energy_feature=energy_feature,
                         is_reference=is_reference,
                         sr=sr, scheduler=scheduler, seed=seed)


    @staticmethod
    def repeat(iterable):
        while True:
            for group in iterable:
                for x in group:
                    yield to_device(x, 'cuda')


    def train(self, step: int) -> torch.Tensor:

        # update model
        self.optimizer.zero_grad()

        # flag for logging
        log_flag = step % self.log_interval == 0

        # forward model
        loss, meta = self.forward(*next(self.train_dataset), is_logging=log_flag)

        # check loss nan
        if loss != loss:
            log('{} cur step NAN is occured'.format(step))
            return

        loss.backward()
        self.clip_grad()
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        # logging
        if log_flag:
            # console logging
            self.console_log('train', meta, step)
            try:
                # tensorboard logging
                self.tensorboard_log('train', meta, step)
            except OverflowError:
                pass


    def run(self):
        train_max_step = self.step + self.max_step
        for i in range(self.step + 1, train_max_step + 1):

            # update step
            self.step = i

            # logging
            if i % self.save_interval == 1:
                log('------------- TRAIN step : %d -------------' % i)

            # do training step
            self.model.train()
            self.train(i)

            # save model
            if i % self.save_interval == 0:
                # save model checkpoint file
                self.save(i)
