import math
import torch
import numpy as np
import lightning as L
import torch.nn as nn
import torch.nn.functional as F

from plot import plot_doverlay, plot_att
from torchvision.transforms.v2 import Compose, Resize
from torchvision.models import resnet18, vit_b_16, ViT_B_16_Weights
from utils import TransformerEncoderLayer_Att, get_2d_sincos_pos_embed


class RegressionModel(L.LightningModule):
    def __init__(self,
                 input_shape: tuple=(2, 256, 256),
                 lr: float=3e-4,
                 weight_decay: float=1e-3,
                 dropout: float=0.0,
                 scheduler: str='cosine',
                 epochs: int = 150,
                 ):
        """
        Base class for regression models.
        Implements train/val/test steps, optimizers, logging, and (vit) plotting

        Args:
            input_shape:
            lr:
            weight_decay:
            dropout:
            scheduler:
            epochs:
        """
        super(RegressionModel, self).__init__()
        self.save_hyperparameters()
        self.test_pred, self.test_labels = [], []
        self.images, self.attn = [], []

    def forward(self, x):
        raise NotImplementedError('RegressionModel superclass cannot be used for direct inference (use subclass)')

    def training_step(self, batch, batch_idx):
        y, x = batch
        y_hat = self.forward(x)
        loss_mse = F.mse_loss(y_hat, y)
        self.log("train_loss", loss_mse, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss_mse

    def validation_step(self, batch, batch_idx):
        y, x = batch
        y_hat = self.forward(x)
        loss_mse = F.mse_loss(y_hat, y)
        loss_mae = F.l1_loss(y_hat, y)

        self.log("val_loss", loss_mse, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_mae", loss_mae, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss_mse

    def test_step(self, batch, batch_idx):
        y, x = batch
        if 'Vit' in self.__class__.__name__ and getattr(self.hparams, 'return_attn', False):
            # saving attention maps
            y_hat, attn = self.forward(x)
            self.images.append(x.detach().cpu())
            self.attn.append(attn.detach().cpu())
        else:
            y_hat = self.forward(x)
        loss_mse = F.mse_loss(y_hat, y)
        loss_mae = F.l1_loss(y_hat, y)

        # Saving test predictions for plotting
        self.test_pred.extend([y_hat.detach().cpu().item()])
        self.test_labels.extend([y.detach().cpu().item()])
        self.log("test_loss", loss_mse, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_mae", loss_mae, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss_mse

    def on_test_end(self):
        tensorboard = self.logger.experiment

        # Plots test predictions
        fig = plot_doverlay(pred=np.array(self.test_pred),
                            GT=np.array(self.test_labels),
                            model=self.__class__.__name__,
                            show=False)
        tensorboard.add_figure('test_predictions', figure=fig)

        # Plots attention maps for ViT models
        if 'Vit' in self.__class__.__name__ and getattr(self.hparams, 'return_attn', False):
            images, attn = torch.cat(self.images, dim=0), torch.cat(self.attn, dim=1)
            idx = np.array(self.test_labels).squeeze().argsort()[
                [int(x) for x in np.linspace(0, len(self.test_labels) - 1, 10,)]
            ]

            for layer in range(attn.shape[0]):
                fig = plot_att(images=images[idx],
                               attention=attn[:, idx, :, :, :],
                               label=list(zip(torch.Tensor(self.test_pred)[idx],
                                              torch.Tensor(self.test_labels)[idx])),
                               layer=layer,
                               show=False)
                tensorboard.add_figure(f'layer {layer} attention scores', figure=fig)

            fig = plot_att(images=images[idx],
                           attention=attn[:, idx, :, :, :],
                           label=list(zip(torch.Tensor(self.test_pred)[idx],
                                          torch.Tensor(self.test_labels)[idx])),
                           layer=None,
                           show=False)
            tensorboard.add_figure(f'average attention scores', figure=fig)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay) ## weight_decay=0.1
        match self.hparams.scheduler:
            case 'cosine':
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.epochs)
            case 'plateau':
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=11)
            case 'exp':
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        return  {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "train_loss"}

    def count_parameters(self):
        return sum([x.numel() for x in self.parameters()])


class LinearModel(RegressionModel):
    def __init__(self,
                 hidden_dim: int,
                 n_layers: int,
                 **kwargs
                 ):
        """
        Simple linear model for regression.

        Args:
            hidden_dim:
            n_layers:
            **kwargs:
        """
        super(LinearModel, self).__init__(**kwargs)
        self.save_hyperparameters()
        C, H, W = self.hparams.input_shape

        layers = [nn.Flatten(-3), nn.Linear(H * W * C, 1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class RegressionMLP(RegressionModel):
    def __init__(self,
                 hidden_dim: int,
                 n_layers: int,
                 **kwargs
                 ):
        """
        Multilayer-Perceptron model for regression.

        Args:
            hidden_dim:
            n_layers:
            **kwargs:
        """
        super(RegressionMLP, self).__init__(**kwargs)
        self.save_hyperparameters()
        C, H, W = self.hparams.input_shape
        dim_in = C * H * W

        layers = [nn.Flatten(-3)]
        for _ in range(self.hparams.n_layers - 1):
            layers.append(nn.Linear(dim_in, self.hparams.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.hparams.dropout))
            dim_in = self.hparams.hidden_dim
        layers.append(nn.Linear(self.hparams.hidden_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class RegressionConv(RegressionModel):
    def __init__(self,
                 in_channels: int=2,
                 num_blocks: int=4,
                 filter_sizes: list=None,
                 use_maxpool: bool=True,
                 **kwargs):
        """
        Convolutional model for regression, implemented as a stack of convolutional blocks.
        Each block doubles the #channels (starts at 32 by default), and halves the spatial dimensions.

        Args:
            in_channels:
            num_blocks:
            filter_sizes:
            use_maxpool:
            **kwargs:
        """
        if filter_sizes is None:
            filter_sizes = [32, 64, 128, 256, 512]
        super(RegressionConv, self).__init__(**kwargs)
        self.save_hyperparameters()
        assert len(filter_sizes) >= num_blocks, "Need at least one filter size per block"

        layers = []
        current_channels = in_channels
        for i in range(num_blocks):
            out_channels = filter_sizes[i]
            layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            if use_maxpool:
                layers.append(nn.MaxPool2d(kernel_size=2))
            current_channels = out_channels

        self.conv_stack = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(current_channels, 2*current_channels),
            nn.ReLU(inplace=True),
            nn.Linear(2*current_channels, 1)
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.global_pool(x)
        x = self.regressor(x)
        return x


class RegressionVit(RegressionModel):
    def __init__(self,
                 patch_size: int,
                 hidden_dim: int,
                 n_heads: int,
                 n_layers: int,
                 mlp_factor: int=4,
                 init: bool=False,
                 return_attn: bool=False,
                 ckpt_path: str=None,
                 **kwargs
                 ):
        """
        Vision Transformer for regression.

        Args:
            patch_size:
            hidden_dim:
            n_heads:
            n_layers:
            mlp_factor:
            init:
            return_attn:
            ckpt_path:
            **kwargs:
        """
        super(RegressionVit, self).__init__(**kwargs)
        self.save_hyperparameters()

        self.x_channels, self.x_height, self.x_width = self.hparams.input_shape
        self.num_patch = self.x_height * self.x_width // self.hparams.patch_size ** 2
        assert self.x_height % self.hparams.patch_size == 0 and self.x_width % self.hparams.patch_size == 0, \
            f'Height ({self.x_height}) and/or Width({self.x_width}) is not devisible by patch_size ({self.hparams.patch_size})'

        self.embedding = nn.Conv2d(in_channels=self.x_channels, out_channels=self.hparams.hidden_dim, kernel_size=self.hparams.patch_size, stride=self.hparams.patch_size)
        self.position_enc = nn.Parameter(
            torch.empty(self.num_patch, self.hparams.hidden_dim).normal_(std=.02), requires_grad=True
        )
        self.transformer = nn.TransformerEncoder(
            TransformerEncoderLayer_Att(d_model=self.hparams.hidden_dim, nhead=self.hparams.n_heads, dropout=self.hparams.dropout,
                                        activation='gelu', batch_first=True, norm_first=True, return_attn=return_attn,
                                        dim_feedforward=self.hparams.mlp_factor*self.hparams.hidden_dim), #dim_feedforward=4*self.hparams.hidden_dim),
            self.hparams.n_layers
        )

        self.output = nn.Sequential(
            nn.Linear(self.hparams.hidden_dim, 2 * self.hparams.hidden_dim),
            nn.GELU(),
            # nn.Dropout(self.hparams.dropout),
            nn.Linear(2 * self.hparams.hidden_dim, 1)
        )
        self.dropout = nn.Dropout(p=self.hparams.dropout)

        if init:
            self.init_params()
        if ckpt_path:
            self.load_ckpt(ckpt_path)


    def forward(self, x):
        # Concat channels x ::
        x = self.embedding(x).reshape(x.shape[0], self.hparams.hidden_dim, -1).permute(0, 2, 1)
        x = self.dropout(x + self.position_enc.repeat(x.shape[0], 1, 1))
        x = self.transformer(x)
        x = self.output(x).mean(dim=1)
        if self.hparams.return_attn:
            scores = []
            for layer in self.transformer.layers:
                scores.append(layer.att_scores)
                layer.att_scores = None
            return x, torch.stack(scores)
        return x

    def init_params(self):
        for param in self.parameters():
            if param.requires_grad and param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def load_ckpt(self, ckpt_path):
        finetune_dict = torch.load(ckpt_path)['state_dict']
        model_dict = self.state_dict()
        with torch.no_grad():
            for name, param in model_dict.items():
                if 'position' in name:
                    # Interpolate Position embeddings if number of patches < number of patches of model
                    num_patch_param = int(math.sqrt(param.shape[0]))
                    if num_patch_param < self.num_patch:
                        param.copy_(F.interpolate(
                            param.unsqueeze(0).reshape(1, num_patch_param, num_patch_param, param.shape[1]).permute(0, 3, 1, 2),
                            size=(int(math.sqrt(self.num_patch)), int(math.sqrt(self.num_patch))),
                            mode='bicubic').squeeze(0).flatten(-2).T
                        )
                        continue
                param.copy_(finetune_dict[name])


class RegressionVit_Conv(RegressionVit):
    def __init__(self,
                 patch_size: int,
                 hidden_dim: int,
                 n_heads: int,
                 n_layers: int,
                 stride: int=None,
                 **kwargs
                 ):
        """
        Regression Vision Transformer with a (overlapping) convolutional embedding layer.

        Args:
            patch_size:
            hidden_dim:
            n_heads:
            n_layers:
            stride:
            **kwargs:
        """
        super().__init__(patch_size, hidden_dim, n_heads, n_layers, **kwargs)
        self.x_channels, self.x_height, self.x_width = self.hparams.input_shape
        self.hparams.stride = stride if stride is not None else self.hparams.patch_size
        embedding, chan_in, chan_out = [], self.x_channels, 24
        out_dim = self.x_height
        while out_dim > (self.x_height // patch_size):
            embedding.append(nn.Conv2d(chan_in, chan_out, 3, 2, 1))
            embedding.append(nn.GELU(inplace=True))
            out_dim = out_dim / 2
            chan_in = chan_out
            chan_out = chan_out * 2
        assert out_dim ** 2 == self.num_patch, f'input height/widht should be multiple of patch_size'
        embedding.append(nn.Conv2d(chan_in, self.hparams.hidden_dim, 1, 1))
        self.embedding = nn.Sequential(*embedding)
        self.position_enc = nn.Parameter(torch.empty(self.num_patch, self.hparams.hidden_dim),
            requires_grad=False
        )


class ResNet(RegressionModel):
    def __init__(self,
                 **kwargs):
        """
        Pretrained ResNet model.
        Adds a regression head and modifies the first embedding layer to use 2 channels.

        Args:
            **kwargs:
        """
        super().__init__(**kwargs)
        self.model = resnet18()
        self.model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3)
        self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=1)

    def forward(self, x):
        return self.model(x)


class MAE(RegressionVit):
    def __init__(self,
                 patch_size: int,
                 hidden_dim: int,
                 n_heads: int,
                 n_layers: int,
                 mlp_factor: int=4,
                 mask_ratio=0.75,
                 **kwargs):
        """
        ViT-based Masked-AutoEncoder model.
        Adds a (ViT-based) decoder and reconstructs patches by minimizing pixel-wise MSE Loss.

        Args:
            patch_size:
            hidden_dim:
            n_heads:
            n_layers:
            mlp_factor:
            mask_ratio:
            **kwargs:
        """
        super().__init__(patch_size, hidden_dim, n_heads, n_layers, **kwargs)
        self.save_hyperparameters()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.hparams.hidden_dim))
        self.decoder_position_enc = nn.Parameter(torch.Tensor(get_2d_sincos_pos_embed(hidden_dim, int(self.x_height // patch_size), False)), False)
        self.decoder = nn.TransformerEncoder(
            TransformerEncoderLayer_Att(d_model=self.hparams.hidden_dim, nhead=self.hparams.n_heads, dropout=self.hparams.dropout,
                                        activation='gelu', batch_first=True, norm_first=True, return_attn=self.hparams.return_attn,
                                        dim_feedforward=mlp_factor*self.hparams.hidden_dim),
            4
        )
        self.output = nn.Linear(self.hparams.hidden_dim, (self.x_channels * patch_size * patch_size))

    def mask_idx(self, im):
        N, N_seq, D = im.shape[0], self.num_patch, self.hparams.hidden_dim
        N_mask = int(N_seq * self.hparams.mask_ratio)

        batch_idx = torch.arange(N, device=self.device).unsqueeze(1)
        idx_random = torch.stack([torch.randperm(N_seq, device=self.device) for _ in range(N)])
        input_idx = idx_random[:, N_mask:].sort(1).values
        masked_idx = idx_random[:, :N_mask].sort(1).values

        patches = self.patchify(im)
        masked_patches = patches[batch_idx, masked_idx]
        return masked_patches, batch_idx, input_idx, masked_idx

    def patchify(self, image):
        """
        Patchifies images.
        Args:
            image: Tensor of shape (B, C, H, W)
        Returns:
            Tensor of shape (B, N_Seq, C*patch_size*patch_size)
        """
        return (image
                .unfold(2, self.hparams.patch_size, self.hparams.patch_size)
                .unfold(3, self.hparams.patch_size, self.hparams.patch_size)
                .permute(0, 2, 3, 1, 4, 5)
                .flatten(1, 2)
                .flatten(-3))

    def depatchify(self, patches):
        """
        Constructs image from patches.
        Args:
            patches: Tensor of shape (B, N_Seq, C*patch_size*patch_size)
        Returns:
            image of shape (B, C, x_height, x_width)
        """
        return F.fold(
            patches.permute(0, 2, 1),
            output_size=(self.x_height, self.x_width),
            kernel_size=(self.hparams.patch_size, self.hparams.patch_size),
            stride=(self.hparams.patch_size, self.hparams.patch_size))


    def forward(self, x):
        # Get GT patches in pixel space and (un)mask indices
        y, batch_idx, input_idx, masked_idx = self.mask_idx(x)
        # Embed and flatten
        x = self.embedding(x).reshape(x.shape[0], self.hparams.hidden_dim, -1).permute(0, 2, 1)
        # Apply pos. embeddings + Dropout
        x = self.dropout(x + self.position_enc.repeat(x.shape[0], 1, 1))
        # Drop Masked patches
        x = x[batch_idx, input_idx]
        # Encoder
        x = self.transformer(x)
        # Fill in mask tokens and add pos encoding
        y_hat = self.mask_token.repeat(x.shape[0], self.num_patch, 1)
        y_hat[batch_idx, input_idx] = x
        y_hat = y_hat + self.decoder_position_enc.repeat(x.shape[0], 1, 1)
        # Decoder
        y_hat  = self.decoder(y_hat)
        # Get masked tokens
        y_hat = y_hat[batch_idx, masked_idx]
        # To pixel space
        y_hat = self.output(y_hat)
        # Saving Attention Maps
        if self.hparams.return_attn:
            scores = []
            for layer in self.transformer.layers:
                scores.append(layer.att_scores)
                layer.att_scores = None
            return y_hat, y, batch_idx, input_idx, masked_idx, torch.stack(scores)
        return y_hat, y, batch_idx, input_idx, masked_idx

    def training_step(self, batch, batch_idx):
        _, x = batch
        y_hat, y, _, _, _ = self.forward(x)
        loss_mse = F.mse_loss(y_hat, y)
        self.log("train_loss", loss_mse, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss_mse

    def validation_step(self, batch, batch_idx):
        _, x = batch
        y_hat, y, _, _, _ = self.forward(x)
        loss_mse = F.mse_loss(y_hat, y)
        loss_mae = F.l1_loss(y_hat, y)

        self.log("val_loss", loss_mse, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_mae", loss_mae, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss_mse

    def test_step(self, batch, batch_idx):
        _, x = batch

        if 'Vit' in self.__class__.__name__ and getattr(self.hparams, 'return_attn', False):
            # saving attention maps
            y_hat, y, _, _, _, attn = self.forward(x)
            self.images.append(x.detach().cpu())
            self.attn.append(attn.detach().cpu())
        else:
            y_hat, y, _, _, _ = self.forward(x)
        loss_mse = F.mse_loss(y_hat, y)
        loss_mae = F.l1_loss(y_hat, y)

        self.log("test_loss", loss_mse, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_mae", loss_mae, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss_mse

    def on_test_end(self):
        tensorboard = self.logger.experiment

        if getattr(self.hparams, 'return_attn', False):
            images, attn = torch.cat(self.images, dim=0), torch.cat(self.attn, dim=1)
            idx = np.random.permutation(len(images))[:10]

            for layer in range(attn.shape[0]):
                fig = plot_att(images=images[idx],
                               attention=attn[:, idx, :, :, :],
                               label=None,
                               layer=layer,
                               show=False)
                tensorboard.add_figure(f'layer {layer} attention scores', figure=fig)

            fig = plot_att(images=images[idx],
                           attention=attn[:, idx, :, :, :],
                           label=None,
                           layer=None,
                           show=False)
            tensorboard.add_figure(f'average attention scores', figure=fig)


class PretrainedVit(RegressionModel):
    def __init__(self,
                 *args,
                 **kwargs):
        """
        Pretrained Vision Transformer model.
        Modifies convolutional embedding layer, adds regression head, and resizes input before processing.
        Args:
            *args:
            **kwargs:
        """
        super().__init__(*args, **kwargs)
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.model.conv_proj = nn.Conv2d(2, 768, kernel_size=16, stride=16)
        for i, (name, param) in enumerate(self.model.named_parameters()):
            if any(string in name for string in ['heads', 'conv', 'encoder.ln', 'encoder_layer_0', 'encoder_layer_10', 'encoder_layer_11', 'class_token', 'pos_embedding']):
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.model.heads = nn.Linear(in_features=768, out_features=1)

        self.transforms = Compose([
            Resize((224, 224)),
        ])

    def forward(self, x):
        return self.model(self.transforms(x))












