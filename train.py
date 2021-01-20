import os, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.model import Model
import hparams as hp
from text import *
from utils.utils import *
from utils.plot_image import *
from apex import amp


def validate(model, val_loader, iteration, writer):
    model.eval()
    with torch.no_grad():
        n_data, val_recon_loss, val_kl_loss, val_duration_loss, val_align_loss = 0, 0, 0, 0, 0
        for i, batch in enumerate(val_loader):
            n_data += len(batch[0])
            text_padded, text_lengths, mel_padded, mel_lengths = [ x.cuda() for x in batch ]
            text_mask, mel_mask, diag_mask = model.prepare_mask(text_lengths, mel_lengths)

            ##### Text #####
            key, value = model.TextEnc(text_padded, text_mask)
            
            ##### Bottom_Up #####
            query=model.bottom_up(mel_padded, mel_mask)

            ##### Alignment #####
            h, align = model.get_align(query, key, value, text_lengths, mel_lengths, text_mask, mel_mask)

            ##### Top_Down #####
            mel_pred, kl_loss = model.top_down(h, mel_mask)

            ##### Compute Loss #####
            duration_out = model.get_duration(value, text_mask)
            recon_loss, duration_loss, align_loss = model.compute_loss(mel_pred,
                                                                       mel_padded,
                                                                       duration_out,
                                                                       align,
                                                                       mel_lengths,
                                                                       text_mask,
                                                                       mel_mask,
                                                                       diag_mask)
            
            val_recon_loss += recon_loss.item() * len(batch[0])
            val_kl_loss += kl_loss.item() * len(batch[0])
            val_duration_loss += duration_loss.item() * len(batch[0])
            val_align_loss += align_loss.item() * len(batch[0])
            
        val_recon_loss /= n_data
        val_kl_loss /= n_data
        val_duration_loss /= n_data
        val_align_loss /= n_data

    writer.add_scalar('val_recon_loss', val_recon_loss, global_step=iteration)
    writer.add_scalar('val_kl_loss', val_kl_loss, global_step=iteration)
    writer.add_scalar('val_duration_loss', val_duration_loss, global_step=iteration)
    writer.add_scalar('val_align_loss', val_align_loss, global_step=iteration)
    
    mel_plots, align_plots = plot_image(mel_padded,
                                        mel_pred,
                                        align,
                                        text_padded,
                                        mel_lengths,
                                        text_lengths)
    writer.add_figure('Validation mel_plots', mel_plots, global_step=iteration)
    writer.add_figure('Validation align_plots', align_plots, global_step=iteration)
    
    mel_out, durations = model.inference(text_padded[-1:, :text_lengths[-1]])
    align = torch.repeat_interleave(torch.eye(len(durations[0].cpu())).to(torch.long),
                                    durations[0].cpu(),
                                    dim=0).unsqueeze(0)
    mel_lengths[-1] = mel_out.size(2)
    mel_plots, align_plots = plot_image(torch.zeros_like(mel_padded),
                                        mel_out,
                                        align,
                                        text_padded,
                                        mel_lengths,
                                        text_lengths)
    writer.add_figure('Validation mel_plots_inference', mel_plots, global_step=iteration)
    writer.add_figure('Validation align_plots_inference', align_plots, global_step=iteration)
    model.train()



def main(args):
    train_loader, val_loader, collate_fn = prepare_dataloaders(hp)
    model = Model(hp).cuda()
    optimizer = torch.optim.Adamax(model.parameters(), lr=hp.lr)
    writer = get_writer(hp.output_directory, args.logdir)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    iteration = 0
    model.train()
    print(f"Training Start!!! ({args.logdir})")
    while iteration < (hp.train_steps):
        for i, batch in enumerate(train_loader):
            text_padded, text_lengths, mel_padded, mel_lengths = [ x.cuda() for x in batch ]
            recon_loss, kl_loss, duration_loss, align_loss = model(text_padded, mel_padded, text_lengths, mel_lengths)

            alpha=min(1, iteration/hp.kl_warmup_steps)
            with amp.scale_loss((recon_loss + alpha*kl_loss + duration_loss + align_loss), optimizer) as scaled_loss:
                scaled_loss.backward()

            iteration += 1
            lr_scheduling(optimizer, iteration)
            nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip_thresh)
            optimizer.step()
            model.zero_grad()
            writer.add_scalar('train_recon_loss', recon_loss, global_step=iteration)
            writer.add_scalar('train_kl_loss', kl_loss, global_step=iteration)
            writer.add_scalar('train_duration_loss', duration_loss, global_step=iteration)
            writer.add_scalar('train_align_loss', align_loss, global_step=iteration)

            if iteration % (hp.iters_per_validation) == 0:
                validate(model, val_loader, iteration, writer)

            if iteration % (hp.iters_per_checkpoint) == 0:
                save_checkpoint(model, optimizer, hp.lr, iteration, filepath=f'{hp.output_directory}/{args.logdir}')

            if iteration == (hp.train_steps):
                break
                
                
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('-d', '--logdir', type=str, required=True)
    args = p.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    torch.manual_seed(hp.seed)
    torch.cuda.manual_seed(hp.seed)
    
    main(args)
