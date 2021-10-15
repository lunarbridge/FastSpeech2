import glob
import os

import torch
import torch.nn as nn
from fastspeech2.trainers.base_trainer import BaseTrainer
from pytorch_sound.utils.commons import get_loadable_checkpoint
from fastspeech2.utils.tools import log
from fastspeech2.utils.tools import to_device
from collections import defaultdict
from pytorch_sound.trainer import LogType


class Evaluator(BaseTrainer):
    def __init__(self, model: nn.Module,
                optimizer, train_dataset, valid_dataset,
                max_step: int, save_interval: int, log_interval: int,
                pitch_feature: str, energy_feature: str,
                save_dir: str, save_prefix: str = '',
                grad_clip: float = 0.0, grad_norm: float = 0.0,
                sr: int = 22050, pretrained_path: str = None, scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                seed: int = 2021, is_reference: bool = False):
        super().__init__(model, optimizer, train_dataset, valid_dataset,
                         valid_max_step=max_step, save_interval=save_interval,
                         log_interval=log_interval, save_dir=save_dir, 
                         save_prefix=save_prefix, grad_clip=grad_clip,
                         grad_norm=grad_norm, pretrained_path=pretrained_path,
                         pitch_feature=pitch_feature, energy_feature=energy_feature,
                         is_reference=is_reference,
                         sr=sr, scheduler=scheduler, seed=seed)

    def validate(self, step: int):
        loss = 0.
        count = 0
        stat = defaultdict(float)

        for i in range(self.valid_max_step):
            # forward model
            with torch.no_grad():
                batch_loss, meta = self.forward(*to_device(next(self.valid_dataset), 'cuda'), is_logging=True)
                loss += batch_loss

            for key, (value, log_type) in meta.items():
                if log_type == LogType.SCALAR:
                    stat[key] += value

            if i % self.log_interval == 0 or i == self.valid_max_step - 1:
                self.console_log('valid', meta, i + 1)

        # averaging stat
        loss /= self.valid_max_step
        for key in stat.keys():
            if key == 'loss':
                continue
            stat[key] = stat[key] / self.valid_max_step
        stat['loss'] = loss

        # update best valid loss
        if loss < self.best_valid_loss:
            self.best_valid_loss = loss

        # console logging of total stat
        msg = 'step {} / total stat'.format(step)
        for key, value in sorted(stat.items()):
            msg += '\t{}: {:.6f}'.format(key, value)
        log(msg)

        # tensor board logging of scalar stat
        for key, value in stat.items():
            self.writer.add_scalar('valid/{}'.format(key), value, global_step=step)
 

    @staticmethod
    def repeat(iterable):
        while True:
            for group in iterable:
                for x in group:
                    yield to_device(x, 'cuda')


    def run(self):
        if not self.pretrained_path:
            raise AttributeError('pretrained path is not set')

        checkpoint_stats = dict()

        check_files = glob.glob(os.path.join(self.pretrained_path, '*'))
        check_files = sorted(check_files, key=os.path.getctime)

        for checkpoint_path in check_files:
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(get_loadable_checkpoint(checkpoint['model']))
            self.model.eval()
            print(f"INFO: load checkpoint '{checkpoint_path}' successfully")

            curr_loss = self.validate(checkpoint['step'])

            # checkpoint_relpath = os.path.relpath(checkpoint_path, self.current_data_path)
            checkpoint_stats[checkpoint_path] = float(curr_loss)

        return checkpoint_stats
