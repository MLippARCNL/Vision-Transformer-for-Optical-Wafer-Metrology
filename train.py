import os
import sys
import lightning as L
sys.path.append(r"C:\Users\dwolf\PycharmProjects\data_analysis_tools")

from data import Data
from pathlib import Path
from utils import LightningCLI_Args
from multiprocessing import freeze_support
from lightning.pytorch.cli import ArgsType
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import RichProgressBar, LearningRateMonitor
from models import (
    RegressionMLP, RegressionConv, RegressionVit, RegressionVit_Conv, ResNet, PretrainedVit, LinearModel, MAE
)


def main(args: ArgsType = None):
    cwd: Path = Path(r'C:\Users\dwolf\PycharmProjects\DeepLearning_DHM_Correction\logs')
    EPOCHS = 250

    # Uncomment to save best & last checkpoint automatically (greatly increases training time)
    # checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(save_top_k=0, save_last=True, every_n_epochs=None, monitor='val_loss', save_weights_only=True, filename='best-{epoch}-{step}')
    # checkpoint_callback.CHECKPOINT_NAME_LAST = 'last-{epoch}-{step}'
    cli = LightningCLI_Args(datamodule_class=Data,
                             args=args,
                             run=True,
                             trainer_defaults=
                             {
                               'max_epochs': EPOCHS,
                               'callbacks': [RichProgressBar(), LearningRateMonitor(logging_interval='epoch')], #checkpoint_callback
                               'enable_checkpointing': False,
                               'precision': '16-mixed',
                               'accelerator': 'gpu', 'devices': 1,
                               'val_check_interval': 1.0,
                               'check_val_every_n_epoch': 1,
                               'log_every_n_steps': 1,
                               'gradient_clip_val': .5,
                               'profiler': None,
                               'reload_dataloaders_every_n_epochs': 0,
                               "logger": {
                                   "class_path": "TensorBoardLogger",
                                   "init_args": {
                                   }
                               }
                             }
                           )

    # Manual checkpoint saving
    ckpt_path = cli.trainer.log_dir + r"\\checkpoints\\last.ckpt"
    cli.trainer.save_checkpoint(ckpt_path)
    model = cli.model
    if 'Vit' in cli.model.__class__.__name__ and hasattr(cli.model.hparams, 'return_attn'):
        args = cli.config_dump['model']['init_args']
        args['return_attn'] = True
        model = cli.model.__class__(**args)
    try:
        out = cli.trainer.test(model=model,
                               datamodule=cli.datamodule,
                               ckpt_path=ckpt_path)
        return out[0]['test_loss']
    except ValueError as v:
        print(v)


if __name__ == "__main__":
    freeze_support()
    main()


