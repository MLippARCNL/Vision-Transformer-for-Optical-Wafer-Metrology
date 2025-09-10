import sys
sys.path.append(r"C:\Users\dwolf\PycharmProjects\data_analysis_tools")
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torch import Tensor
from math import ceil, floor

from matplotlib.figure import Figure
from matplotlib import colors


def plot_dataset(images, labels, show=False, cbar='global', show_ROI=False):
    """
    Plots the images from the dataset.
    (Use N_images=10)
    Args:
        images: Tensor (N, C, H, W)
        labels: Tensor (N, )
        show: True=Show plot
        cbar: global=global colorbar (otherwise individual bars per image)
        show_ROI: True=Show Region of Interest (ROI)
    """
    N, C, H, W = images.shape
    fig, axs = plt.subplots(C, N, sharex=True, sharey=True, constrained_layout=True, figsize=(20, 20 // (N / C)))
    if cbar == 'global':
        norm = colors.Normalize(vmin=torch.min(images), vmax=torch.max(images))
        for i in range(images.shape[0]):
            im1 = axs[0, i].imshow(images[i][0], norm=norm)
            im2 = axs[1, i].imshow(images[i][1], norm=norm)
            axs[0, i].title.set_text(f"OV: {labels[i].item():.4f}")
        fig.colorbar(im2, ax=axs, orientation='vertical', pad=0.01)
    else:
        for i in range(images.shape[0]):
            fig.colorbar(axs[0, i].imshow(images[i][0]), ax=axs[0, i], orientation='vertical')
            fig.colorbar(axs[1, i].imshow(images[i][1]), ax=axs[1, i], orientation='vertical')
            axs[0, i].title.set_text(f"OV: {labels[i].item():.4f}")
    if show_ROI:
        for i in range(N):
            rect_1 = patches.Rectangle((40, 40), 80, 80, linewidth=2, edgecolor='r', facecolor='none')
            rect_2 = patches.Rectangle((130, 135), 80, 80, linewidth=2, edgecolor='r', facecolor='none')
            axs[0, i].add_patch(rect_1)
            axs[0, i].add_patch(rect_2)

            rect_1 = patches.Rectangle((40, 40), 80, 80, linewidth=2, edgecolor='r', facecolor='none')
            rect_2 = patches.Rectangle((130, 135), 80, 80, linewidth=2, edgecolor='r', facecolor='none')
            axs[1, i].add_patch(rect_1)
            axs[1, i].add_patch(rect_2)

            delta_plus = (images[i][1][40:120, 40:120].mean() - images[i][0][40:120, 40:120].mean())
            delta_neg = (images[i][1][130:210, 135:215].mean() - images[i][0][130:210, 135:215].mean())
            axs[1, i].title.set_text(f"ROI OV: {(-20 * ((delta_plus + delta_neg) / (delta_plus - delta_neg))):.4f}")
    plt.suptitle('±1 diffraction orders amplitude maps for varying overlay (OV)', fontsize=20)
    if show:
        plt.show()
    return fig

def plot_doverlay(pred, GT, model, show=False):
    """
    Plot predictions against ground truth.
    Plots Dov on y axis (GT - pred).
    Args:
        pred: Tensor (N, )
        GT: Tensor (N, )
        model: string (name)
        show: bool=True (show plot)
    """
    delta_ov = pred - GT
    max_dev_y, min_dev_y = max(delta_ov), min(delta_ov)
    max_dev_x, min_dev_x = GT[np.where(delta_ov == max_dev_y)], GT[np.where(delta_ov == min_dev_y)]
    mean_dev_y = np.mean(delta_ov)

    fig = plt.figure(figsize=(10, 6))

    plt.scatter(GT, delta_ov, alpha=0.5, s=0.2, label='Predictions')
    plt.hlines(0, floor(min(GT)), ceil(max(GT)) + 1, colors='r', label='Perfect Prediction')
    plt.hlines(mean_dev_y, floor(min(GT)), ceil(max(GT)) + 1, colors='b', linestyles='dashed', label='Mean Deviation')
    plt.vlines(x=(max_dev_x, min_dev_x), ymin=(0, min_dev_y), ymax=(max_dev_y, 0), colors='r', linestyles='dashed',
               label='Max Positive/Negative ΔOV')

    plt.text(max_dev_x, max_dev_y, f'ΔOV: ({max_dev_y:.2f})', color='red', fontsize=10, verticalalignment='bottom',
             horizontalalignment='right')
    plt.text(min_dev_x, min_dev_y, f'ΔOV: ({min_dev_y:.2f})', color='red', fontsize=10, verticalalignment='top',
             horizontalalignment='right')
    plt.text(ceil(max(GT)), mean_dev_y, f'ΔOV: {mean_dev_y:.2f}', color='blue', fontsize=10,
             verticalalignment='bottom', horizontalalignment='left')

    half = np.linspace(0, ceil(max(abs(min_dev_y), abs(max_dev_y))), 10)
    plt.yticks(np.append(-np.flip(half), half))
    plt.xticks(np.arange(floor(min(GT)), ceil(max(GT) + 1), 1))
    plt.xlabel('OV (nm)')
    plt.ylabel('ΔOV (nm)')
    plt.title(f'{model} Overlay predictions')
    plt.legend(markerscale=20.)

    if show:
        plt.show()
    return fig


def fig_5(df, show=False):
    """
    Plot predictions against ground truth.
    Plots Dov on y axis (GT - pred).
    Args:
        pred: Tensor (N, )
        GT: Tensor (N, )
        model: string (name)
        show: bool=True (show plot)
    """
    for target in ['16', '10']:
        w1_pred, w2_pred = np.array(eval(df[(df.Dataset == f'W1_{target}')]['pred'].array[0])), np.array(
            eval(df[(df.Dataset == f'W2_{target}')]['pred'].array[0]))
        w1_GT, w2_GT = np.array(eval(df[(df.Dataset == f'W1_{target}')]['GT'].array[0])), np.array(
            eval(df[(df.Dataset == f'W2_{target}')]['GT'].array[0]))
        delta_ov_1 = w1_pred - w1_GT
        delta_ov_2 = w2_pred - w2_GT

        fig = plt.figure(figsize=(10, 6))
        plt.scatter(w1_GT, delta_ov_1, alpha=1, s=0.8, label='Wavefront 1')
        plt.scatter(w2_GT, delta_ov_2, alpha=1, s=0.8, label='Wavefront 2')
        half = np.linspace(0, .900, 5)
        plt.yticks(np.append(-np.flip(half), half))
        plt.xticks(np.arange(floor(min(w1_GT)), ceil(max(w1_GT) + 1), 1))
        plt.xlabel('OV (nm)')
        plt.ylabel('ΔOV (nm)')
        plt.title(f' Overlay predictions')
        plt.legend(markerscale=10.)

        if show:
            plt.show()
    return fig



def plot_heads(images, attention, show=False):
    """
    Plots attention heads
    Args:
        images: Tensor (N, C, H, W)
        attention: Tensor (L, N, H, N_Seq, D)
        show: bool=True (show plot)
    """
    N = images.shape[0]
    L, _, H, N_Seq, _ = attention.shape
    token_scores = attention.sum(dim=3).reshape(L, N, H, int(math.sqrt(N_Seq)), int(math.sqrt(N_Seq)))
    # maps = nn.Upsample(size=(256, 256))(token_scores)
    for n in range(N):
        fig, axs = plt.subplots(L, H, figsize=(H*2, H*2/(H//L)), sharex=True, sharey=True, constrained_layout=True)
        maps = nn.Upsample(size=(256, 256), mode='bilinear')(token_scores[:, n, :, :, :])
        for l in range(L):
            norm = colors.Normalize(vmin=torch.min(token_scores[l, n, :, :, :]),
                                    vmax=torch.max(token_scores[l, n, :, :, :]))
            for h in range(H):
                im = axs[l, h].imshow(maps[l][h], norm=norm)
            fig.colorbar(im, ax=axs[l, :])
        if show:
            fig.show()

def plot_att(images: Tensor, attention: Tensor, layer: int = None, label: list[tuple[Tensor, Tensor]] = None, show=False
             ) -> Figure:
    """
    Plots the attention scores for each image in images
    Averages over each of the attention heads
    Args:
        images: Tensor (N, C, H, W)
        attention: Tensor (L, N, H, N_Seq, D)
        layer: int
        label: list[tuple[Tensor, Tensor]]  (pred, gt)
        show: bool=True (show plot)
    """
    title = 'Attention scores for layer {layer}'.format(layer=layer)
    if layer is None:
        attention = attention.mean(dim=0).unsqueeze(0)
        layer = 0
        title = 'Attention scores averaged over all layers'

    N = images.shape[0]
    _, H, N_Seq, _ = attention[layer].shape
    Patch_dim = int(np.sqrt(N_Seq))
    att_map = attention[layer].mean(dim=1)
    token_scores = att_map.sum(dim=1)

    fig, axs = plt.subplots(3, N, figsize=(N*2, N*2/(N//3)), sharex=True, sharey=True, constrained_layout=True)
    norm1 = colors.Normalize(vmin=torch.min(token_scores), vmax=torch.max(token_scores))
    norm2 = colors.Normalize(vmin=torch.min(images), vmax=torch.max(images))
    for n in range(N):
        maps = token_scores[n].reshape(1, 1, Patch_dim, Patch_dim)
        maps = nn.Upsample(size=(256, 256), mode='bilinear')(maps)
        im2 = axs[2, n].imshow(maps[0, 0, :, :], norm=norm1, cmap='viridis')
        im0 = axs[0, n].imshow(images[n][0], norm=norm2)
        im1 = axs[1, n].imshow(images[n][0], norm=norm2)

        if label is not None:
            pred, gt = label[n]
            axs[0, n].title.set_text(f"GT: {gt.item():.4f}")
            axs[1, n].title.set_text(f"PRED: {pred.item():.4f}")

    fig.colorbar(im1, ax=axs[:2, :], location='right', shrink=0.7, orientation='vertical',
                         format=lambda x, pos: '{:.1f}'.format(x), pad=0.01)
    fig.colorbar(im2, ax=axs[2:3, :], location='right', shrink=0.9, orientation='vertical',
                         format=lambda x, pos: '{:.1f}'.format(x), pad=0.01)
    fig.suptitle(title, fontsize=20)
    if show:
        plt.show()
    return fig

def plot_datascaling(df, dataset='hard_10', show=False):
    fig = plt.figure()
    for model, df_model in df[df['Dataset'] == dataset].groupby('Model'):
        df_model['Date_id'] = df_model['Date'].str.extract('(\d+)', expand=False).astype(int)
        df_model = df_model.sort_values('Date_id')
        y = df_model.groupby('Date_id')['test_loss'].mean().apply(lambda x: x[0])
        y_err = df_model.groupby('Date_id')['test_loss'].std()
        x = [x.split('=')[-1] for x in df_model['Date'].unique()]
        plt.errorbar(x, y, label=model)
        plt.fill_between(x, np.subtract(y, y_err), np.add(y, y_err), alpha=0.2)
    plt.title(f"Dataset: {dataset}")
    plt.xlabel('MSE Loss')
    plt.ylabel('Budget')
    plt.legend()
    if show:
        plt.show()
    else:
        return fig

def plot_datascaling2(df, dataset='medium_10'):
    fig = plt.figure()
    for model, df_model in df[df['Dataset'] == dataset].groupby('ModelType'):
        df_model['Budget'] = df_model['Budget'].apply(lambda x: int(x))
        df_model = df_model.sort_values('Budget')
        df_model['val_loss-1'] = df_model['val_loss'].apply(lambda x: x[-1])
        y = df_model.groupby('Budget')['val_loss-1'].mean()
        y = y.reindex(sorted(y.index))
        y_err = df_model.groupby('Budget')['val_loss-1'].std()
        y_err = y_err.reindex(sorted(y_err.index))
        x = [str(x) for x in df_model['Budget'].unique()]
        plt.errorbar(x, y, label=model)
        plt.fill_between(x, np.subtract(y, y_err), np.add(y, y_err), alpha=0.2)
    plt.title(f"Dataset: {dataset}")
    plt.xlabel('Budget')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

def mean_metric_over_wavefronts(df, metric='test_mae'):
    df['Target'] = df['Dataset'].str.split('_').apply(lambda x: f'C{x[-1]}')
    df['Dataset'] = df['Dataset'].str.split('_').apply(lambda x:
                                                      'Wavefront 0' if x[0] == 'W0' else
                                                      'Wavefront 1' if x[0] == 'W1' else
                                                      'Wavefront 2')
    df[metric] = df[metric].apply(lambda x: x[-1])
    df = df.groupby(['Target', 'Dataset'])[metric].mean()

    fig, ax = plt.subplots(figsize=(6, 4))
    yaxis = df.index.get_level_values(1).unique().to_numpy()
    x = np.arange(len(yaxis))
    width = 0.35

    bars1 = ax.bar(x - width / 2, df.loc[['C16']].values, width, label='C16', color='steelblue')
    bars2 = ax.bar(x + width / 2, df.loc[['C10']].values, width, label='C10', color='coral')
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12)

    ax.set_ylabel('Mean Absolute Error (MAE)')
    ax.set_ylim([0.00, 0.25])
    ax.set_xticks(x)
    ax.set_xticklabels(yaxis)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(title='Target')
    plt.grid(alpha=.7, axis='both', linestyle='--')

    plt.tight_layout()
    plt.show()




