import matplotlib.pyplot as plt
import torch.nn.functional as F
from text.symbols import symbols
import hparams as hp


# Mappings from symbol to numeric ID and vice versa:
symbol_to_id = {s: i for i, s in enumerate(symbols)}
id_to_symbol = {i: s for i, s in enumerate(symbols)}



def plot_image(target, melspec, alignments, text, mel_lengths, text_lengths):
    # Draw mel_plots
    mel_plots, axes = plt.subplots(2,1,figsize=(20,15))
    L, T = text_lengths[-1], mel_lengths[-1]

    axes[0].imshow(target[-1].detach().cpu()[:,:T],
                   origin='lower',
                   aspect='auto')

    axes[1].imshow(melspec[-1].detach().cpu()[:,:T],
                   origin='lower',
                   aspect='auto')

    # Draw alignments
    align_plots, axes = plt.subplots(2,1,figsize=(20,15))
    alignments = alignments[-1].repeat_interleave(int(hp.downsample_ratio),0).t()
    alignments = alignments.detach().cpu()[:L,:T]
    
    axes[0].imshow(alignments,
                   origin='lower',
                   aspect='auto')

    _, alignments = alignments.max(dim=0)
    alignments = F.one_hot(alignments, L).t()
    axes[1].imshow(alignments,
                   origin='lower',
                   aspect='auto')
    
    for i in range(2):
        plt.sca(axes[i])
        plt.xticks(range(T), [ f'{i}' if (i%10==0 or i==T-1) else '' for i in range(T) ])
        plt.yticks(range(L), [id_to_symbol[c] for c in text[-1].detach().cpu().numpy()[:L]])
        for yc in range(L):
            plt.axhline(y=yc, c='r', linestyle='--', linewidth=0.5)
    
    return mel_plots, align_plots