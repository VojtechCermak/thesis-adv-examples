import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
import torch.nn.functional as F
from torchvision.utils import make_grid


def interpolation_plot(imgs, classifier, original_class, target_class, xticks, legend, xlabel=None, ylabel=None):
    '''
    Inputs sequence of images and classifier and plots probabilities. 
    '''
    # Data: Probabilities
    preds = classifier(imgs)
    preds = F.softmax(preds, dim=1).cpu().detach().numpy()

    pred_original = preds[:, original_class]
    pred_target = preds[:, target_class]
    pred_other = np.sum(np.delete(preds, [original_class, target_class], axis=1), axis=1)
    data = [pred_original, pred_target, pred_other]

    # Data: Interpolated images grid
    grid = make_grid(imgs, nrow=36, normalize=True)
    grid = grid.cpu().detach().numpy()
    grid = np.transpose(grid, (1,2,0))

    # Plot
    fig = plt.figure(figsize=(15, 2))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(grid)
    ax1.set_axis_off()

    ax2 = fig.add_subplot(gs[1])
    ax2.margins(x=0, y=0)
    ax2.stackplot(np.arange(len(preds)), data, labels=legend, colors=sns.color_palette("Pastel1", 3))
    ax2.set_xticks(np.linspace(0, len(preds)-1, len(xticks)))
    ax2.set_xticklabels(xticks)

    if xlabel is None:
        xlabel = 'Step'
    ax2.set_xlabel(xlabel)

    if ylabel is None:
        ylabel = 'Softmax of class'
    ax2.set_ylabel(ylabel)

    ax2.legend(prop={'size': 8})
    return fig


def plot_single_sample(ax, i, rows, cols, data, labelfont=8, grayscale=False):
    img_normalized = (data['imgs'][i] / 2 + 0.5)
    img = np.moveaxis(img_normalized, 0, -1)
    lt = data['labels_target'][i]
    lo = data['labels_origin'][i]

    ax.tick_params(axis=u'both', which=u'both',length=0, labelsize=labelfont)
    if i % cols == 0:
        ax.set_yticks([img.shape[0] //2])
        ax.set_yticklabels([f"{data['class_map'][lo]}"])
    else:
        ax.set_yticks([])
    if (rows // 2) * cols == i:
        ax.set_ylabel('Classified by human')
    if (cols*rows) - (cols // 2) - 1  == i:
        ax.set_xlabel('Classified by machine')

    ax.set_xticks([img.shape[1] //2])
    ax.set_xticklabels([f"{data['class_map'][lt]} - {data['preds'][i][lt]:.3f}"])
    if grayscale:
        ax.imshow(img, cmap='gray')
    else:
        ax.imshow(img)
    return ax


def plot_matrix(ax, matrix, title, textcolors=("black", "white")):
    im = ax.imshow(matrix, cmap="magma_r")
    ax.set_title(title, size=14, y=1.1)

    # Set x axis
    ax.set_xlabel(matrix.columns.name)    
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position('top')
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)

    # Set y axis
    ax.set_ylabel(matrix.index.name)  
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)

    # Loop over data dimensions and create text annotations.
    threshold = im.norm(matrix.values.max())/2.
    for j in range(len(matrix.columns)):
        for i in range(len(matrix.index)):
            color = textcolors[int(im.norm(matrix.values[i, j]) > threshold)]
            text = ax.text(j, i, f"{matrix.values[i, j]:.1f}", ha="center", va="center", color=color)
