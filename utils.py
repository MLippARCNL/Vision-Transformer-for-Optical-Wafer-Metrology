import os

import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from datetime import date
from lightning.pytorch.cli import LightningCLI
from tensorboard.backend.event_processing import event_accumulator


class TransformerEncoderLayer_Att(nn.TransformerEncoderLayer):
    def __init__(self,
                 return_attn=False,
                 **kwargs):
        """
        Subclass of nn.TransformerEncoderLayer that returns attention scores by overriding the forward method
        Stores the last computed attention scores in self.att_scores
        Args:
            return_attn:
            **kwargs:
        """
        super(TransformerEncoderLayer_Att, self).__init__(**kwargs)
        self.return_attn = return_attn
        if self.return_attn:
            self.att_scores = None

    def forward(
        self, src, src_mask=None, src_key_padding_mask=None, is_causal=False,
    ):
        """
        Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``src mask``.
                Default: ``False``.
                Warning:
                ``is_causal`` provides a hint that ``src_mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype,
        )
        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        x = src
        if self.norm_first:
            x_hat, att = self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal
            )
            x = x + x_hat
            x = x + self._ff_block(self.norm2(x))
        else:
            x_hat, att = self._sa_block(
               x, src_mask, src_key_padding_mask, is_causal=is_causal
            )
            x = self.norm1(
                x +  x_hat
            )
            x = self.norm2(x + self._ff_block(x))

        if self.return_attn:
            self.att_scores = att

        return x

    # self-attention block
    def _sa_block(
        self, x, attn_mask, key_padding_mask, is_causal=False,
    ):
        x, att = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
            is_causal=is_causal,
        )
        return self.dropout1(x), att


class LightningCLI_Args(LightningCLI):
    """
    CLI with adjusted arguments and logging.
    Adds --name (Experiment_name), --type (Model Type)
    Adjusts the savedir s.t. savedir= //--name//--model//--data.train_file//--model.type//version
    --> See run_multiple
    """
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--name", default="Testing", type=str)
        parser.add_argument("--type", default="None", type=str)
        parser.link_arguments("name", "trainer.logger.init_args.save_dir", compute_fn=lambda x: "logs/{}".format(x))
        parser.link_arguments(("model", "data.train_file", "type"), "trainer.logger.init_args.name",
                              compute_fn=lambda x, y, z: x.class_path.split(".")[-1] + f"/{y}/{date.today().strftime('%Y-%m-%d')}"
                              if z == "None" else x.class_path.split(".")[-1] + f"/{y}/{z}" )
        parser.link_arguments("trainer.max_epochs", "model.init_args.epochs")


def parse_tensorboard(models_path, scalars=['train_loss', 'val_loss', 'test_loss', 'val_mae', 'test_mae']):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    df = []
    for name in os.listdir(models_path):
        model_path = os.path.join(models_path, name)

        for dataset in os.listdir(model_path):
            dataset_path = os.path.join(model_path, dataset)

            for date in os.listdir(dataset_path):
                date_path = os.path.join(dataset_path, date)

                for version in os.listdir(os.path.join(model_path, dataset, date)):
                    version_path = os.path.join(date_path, version)
                    row = {'Model': name, 'Dataset': dataset, 'ModelType': date, 'Version': version}

                    for file in os.listdir(version_path):

                        if file.startswith('events.out.tfevents'):
                            file_path = os.path.join(version_path, file)
                            ea = event_accumulator.EventAccumulator(
                                file_path,
                                size_guidance={event_accumulator.SCALARS: 0},
                            )

                            try:
                                ea.Reload()
                                for scalar in set(ea.scalars._buckets.keys()).intersection(set(scalars)):
                                    row[scalar] = pd.DataFrame(ea.Scalars(scalar))['value'].to_numpy()

                            except Exception as e:
                                print("Exception while parsing {}".format(file))

                    df.append(row)

    return pd.DataFrame(df)

import numpy as np
# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

if __name__ == "__main__":
    parse_tensorboard('logs/TestRuns-Augmented/RegressionConv')