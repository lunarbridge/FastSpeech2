from typing import Dict, Tuple

import torch
import torch.nn as nn
from fastspeech2.models.loss import FastSpeech2Loss
from fastspeech2.utils.tools import to_device
from pytorch_sound.settings import MIN_DB
from pytorch_sound.trainer import LogType, Trainer
# from pytorch_sound.utils.tensor import to_device
from pytorch_sound.utils.calculate import db2log
from speech_interface.interfaces.hifi_gan import InterfaceHifiGAN


class BaseTrainer(Trainer):
    def __init__(self, model: nn.Module,
                 optimizer, train_dataset, valid_dataset,
                 save_interval: int, log_interval: int,
                 pitch_feature: str, energy_feature: str,
                 save_dir: str, save_prefix: str = '',
                 max_step: int = 0, valid_max_step: int = 0,
                 grad_clip: float = 0.0, grad_norm: float = 0.0,
                 sr: int = 22050, pretrained_path: str = None, scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                 seed: int = 2021, is_reference: bool = False):
        super().__init__(model, optimizer, train_dataset, valid_dataset,
                         max_step, valid_max_step, save_interval, log_interval, save_dir, save_prefix,
                         grad_clip, grad_norm, pretrained_path, sr=sr, scheduler=scheduler, seed=seed)
        # vocoder
        self.interface = InterfaceHifiGAN(
            model_name='hifi_gan_v1_universal', device='cuda'
        )

        # make loss
        self.loss_func = FastSpeech2Loss(pitch_feature, energy_feature)

        self.is_reference = is_reference
        self.mel_log_min = db2log(MIN_DB)

    def forward(self, *inputs, is_logging: bool = False) -> Tuple[torch.Tensor, Dict]:
        # Forward
        output = self.model(*inputs[2:])

        # calculate loss
        losses = self.loss_func(inputs, output)
        loss = losses[0]  # total loss

        if is_logging:
            id_, text = inputs[:2]
            total_loss, mel_loss, post_loss, pitch_loss, energy_loss, duration_loss = losses

            raugh_mel, post_mel = output[:2]
            raugh_mel, post_mel = raugh_mel[:1].transpose(1, 2), post_mel[:1].transpose(1, 2)
            target_mel = inputs[6][:1].transpose(1, 2)

            # slice padded part, minimum value of mel is zero.
            if any([self.mel_log_min + 1 > numb for numb in target_mel[0, 0].cpu().numpy().tolist()]):
                first_pad_idx = int(target_mel[0, 0].argmin().cpu().numpy())
                raugh_mel = raugh_mel[..., :first_pad_idx]
                post_mel = post_mel[..., :first_pad_idx]
                target_mel = target_mel[..., :first_pad_idx]

            # synthesis
            pred_wav = self.interface.decode(post_mel).squeeze()
            rec_wav = self.interface.decode(target_mel).squeeze()
            raugh_mel, post_mel, target_mel = raugh_mel[0], post_mel[0], target_mel[0]

            meta = {
                # losses
                'total_loss': (total_loss.item(), LogType.SCALAR),
                'mel_loss': (mel_loss.item(), LogType.SCALAR),
                'post_loss': (post_loss.item(), LogType.SCALAR),
                'pitch_loss': (pitch_loss.item(), LogType.SCALAR),
                'energy_loss': (energy_loss.item(), LogType.SCALAR),
                'duration_loss': (duration_loss.item(), LogType.SCALAR),
                # plots
                'mel.target': (target_mel, LogType.IMAGE),
                'mel.raugh': (raugh_mel, LogType.IMAGE),
                'mel.post': (post_mel, LogType.IMAGE),
                'wav.target.plot': (rec_wav, LogType.PLOT),
                'wav.target.audio': (rec_wav, LogType.AUDIO),
                'wav.pred.plot': (pred_wav, LogType.PLOT),
                'wav.pred.audio': (pred_wav, LogType.AUDIO),
                # text
                'id_': (id_[0], LogType.TEXT),
                'text': (text[0], LogType.TEXT)
            }
        else:
            meta = {}
        return loss, meta

    @staticmethod
    def repeat(iterable):
        while True:
            for group in iterable:
                for x in group:
                    yield to_device(x, 'cuda')
