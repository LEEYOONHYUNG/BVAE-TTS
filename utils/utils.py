import os
import torch
from torch.utils.data import DataLoader
from .data_utils import TextMelSet, TextMelCollate
from torch.utils.tensorboard import SummaryWriter
import hparams as hp


def prepare_dataloaders(hp):
    # Get data, data loaders and collate function ready
    trainset = TextMelSet(hp.training_files, hp)
    valset = TextMelSet(hp.validation_files, hp)
    collate_fn = TextMelCollate()

    train_loader = DataLoader(trainset,
                              num_workers=4,
                              shuffle=True,
                              batch_size=hp.batch_size, 
                              drop_last=True, 
                              collate_fn=collate_fn)
    
    val_loader = DataLoader(valset,
                            num_workers=4,
                            batch_size=hp.batch_size,
                            collate_fn=collate_fn)
    
    return train_loader, val_loader, collate_fn


def get_writer(output_directory, log_directory):
    logging_path=f'{output_directory}/{log_directory}'
    if not os.path.exists(logging_path):
        os.mkdir(logging_path)
    writer = SummaryWriter(logging_path)
    return writer


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print(f"Saving model and optimizer state at iteration {iteration} to {filepath}")
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, f'{filepath}/checkpoint_{iteration}')

    
def lr_scheduling(opt, step, init_lr=hp.lr, warmup_steps=hp.lr_warmup_steps):
    opt.param_groups[0]['lr'] = init_lr * warmup_steps**0.5 * min(step ** -0.5, step * warmup_steps ** -1.5)
    return


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = lengths.new_tensor(torch.arange(0, max_len, device=lengths.device))
    mask = (lengths.unsqueeze(1) <= ids).to(torch.bool)
    return mask.detach()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def PositionalEncoding(d_model, lengths, w_s=None):
    L = int(lengths.max().item())
    if w_s is None:
        position = torch.arange(0, L, dtype=torch.float).unsqueeze(0).to(lengths.device)
    else:
        position = torch.arange(0, L, dtype=torch.float).unsqueeze(0).to(lengths.device) * w_s.unsqueeze(-1)
    div_term = torch.pow(10000, torch.arange(0, d_model, 2).float() / d_model).to(lengths.device)
    pe = torch.zeros(len(lengths), L, d_model).to(lengths.device)
    
    pe[:, :, 0::2] = torch.sin(position.unsqueeze(-1) / div_term.unsqueeze(0))
    pe[:, :, 1::2] = torch.cos(position.unsqueeze(-1) / div_term.unsqueeze(0))
    return pe
