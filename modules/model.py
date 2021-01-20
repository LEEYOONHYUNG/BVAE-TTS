import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
from .conv import *
from utils.utils import *


class Model(nn.Module):
    def __init__(self, hp):
        super(Model, self).__init__()
        self.hp=hp
        self.ratio=hp.downsample_ratio
        self.text_mask=None
        self.mel_mask=None
        self.diag_mask = None

        # build network
        self.Prenet = Prenet(hp)
        self.TextEnc = TextEnc(hp)
        
        self.BVAE_blocks = nn.ModuleList()
        for i in range(hp.n_blocks):
            ForT= 'F' if i%2==0 else 'T'
            self.BVAE_blocks.append(BVAE_block(hp.hidden_dim//2**(i//2+1),
                                               hp.kernel_size,
                                               hp.n_layers,
                                               down_upsample=ForT))
            
        self.Query = Conv1d(hp.hidden_dim//self.ratio, hp.hidden_dim, hp.kernel_size, bias=False)
        self.Compress = Linear(hp.hidden_dim, hp.hidden_dim//self.ratio, bias=False)
        self.Projection = Projection(hp.hidden_dim, hp.kernel_size, hp.n_mel_channels)
        
        # duration predictor
        self.Duration = DurationPredictor(hp)

        
    def forward(self, text, melspec, text_lengths, mel_lengths):
        ##### Prepare Mask#####
        self.text_mask, self.mel_mask, self.diag_mask = self.prepare_mask(text_lengths, mel_lengths)
        
        ##### Text #####
        key, value = self.TextEnc(text, self.text_mask)
        
        ##### Bottom_Up #####
        query=self.bottom_up(melspec, self.mel_mask)
        
        ##### Alignment #####
        h, align = self.get_align(query, key, value, text_lengths, mel_lengths, self.text_mask, self.mel_mask)

        ##### Top_Down #####
        mel_pred, kl_loss = self.top_down(h, self.mel_mask)
        
        ##### Compute Loss #####
        duration_out = self.get_duration(value, self.text_mask)
        recon_loss, duration_loss, align_loss = self.compute_loss(mel_pred,
                                                                  melspec,
                                                                  duration_out,
                                                                  align,
                                                                  mel_lengths,
                                                                  self.text_mask,
                                                                  self.mel_mask,
                                                                  self.diag_mask)
        
        return recon_loss, kl_loss, duration_loss, align_loss
    
    
    def prepare_mask(self, text_lengths, mel_lengths):
        B, L, T = text_lengths.size(0), text_lengths.max().item(), mel_lengths.max().item()
        text_mask = get_mask_from_lengths(text_lengths)
        mel_mask = get_mask_from_lengths(mel_lengths)
        x = (torch.arange(L).float().unsqueeze(0).to(text_lengths.device)/text_lengths.unsqueeze(1)).unsqueeze(1)\
             - (torch.arange(T//self.ratio).float().unsqueeze(0).to(text_lengths.device)/(mel_lengths//self.ratio).unsqueeze(1)).unsqueeze(2)
        diag_mask = (-12.5*torch.pow(x, 2)).exp()
        diag_mask = diag_mask.masked_fill(text_mask.unsqueeze(1), 0)
        diag_mask = diag_mask.masked_fill(mel_mask[:,::self.ratio].unsqueeze(-1), 0)
        
        return text_mask, mel_mask, diag_mask
        
        
    def bottom_up(self, melspec, mel_mask):
        x = self.Prenet(melspec, mel_mask)
        for i, block in enumerate(self.BVAE_blocks):
            x = block.up(x, mel_mask[:, ::2**((i+1)//2)])
        
        query = self.Query(x, mel_mask[:,::self.ratio]).transpose(1,2)
        
        return query
    
    
    def top_down(self, h, mel_mask):
        kl = 0
        for i, block in enumerate(reversed(self.BVAE_blocks)):
            h, curr_kl = block.down(h, mel_mask[:, ::2**(len(self.BVAE_blocks)//2-(i+1)//2)])
            kl += curr_kl
            
        mel_pred = self.Projection(h, mel_mask)
        
        return mel_pred, kl

    
    def get_align(self, q, k, v, text_lengths, mel_lengths, text_mask, mel_mask):
        q = q + PositionalEncoding(self.hp.hidden_dim, mel_lengths//self.ratio)
        k = k + PositionalEncoding(self.hp.hidden_dim, text_lengths, 1.0*mel_lengths/self.ratio/text_lengths)
        
        q = q * self.hp.hidden_dim ** -0.5
        scores = torch.bmm(q, k.transpose(1, 2))
        scores = scores.masked_fill(text_mask.unsqueeze(1), -float('inf'))
        
        align = scores.softmax(-1)
        align = align.masked_fill(mel_mask[:,::self.ratio].unsqueeze(-1), 0)
        if self.training:
            align_oh = self.jitter(F.one_hot(align.max(-1)[1], align.size(-1)), mel_lengths)
        else:
            align_oh = F.one_hot(align.max(-1)[1], align.size(-1))
        align_oh = align_oh.masked_fill(mel_mask[:,::self.ratio].unsqueeze(-1), 0)
        
        attn_output = torch.bmm(align + (align_oh-align).detach(), v)
        attn_output = self.Compress(attn_output).transpose(1,2)
        
        return attn_output, align
    
       
    def compute_loss(self, mel_pred, mel_target, duration_out, align, mel_lengths, text_mask, mel_mask, diag_mask):
        # Recon Loss
        recon_loss = nn.L1Loss()(mel_pred.masked_select(~mel_mask.unsqueeze(1)),
                                 mel_target.masked_select(~mel_mask.unsqueeze(1)))

        # Duration Loss
        duration_target = self.align2duration(align, mel_lengths)
        duration_target_flat = duration_target.masked_select(~text_mask)
        duration_target_flat[duration_target_flat<=0]=1
        duration_out_flat = duration_out.masked_select(~text_mask)
        duration_loss = nn.MSELoss()( torch.log(duration_out_flat+1e-5), torch.log(duration_target_flat+1e-5) )
        
        # Guide Loss
        align_losses = align*(1-diag_mask)
        align_loss = torch.mean(align_losses.masked_select(diag_mask.bool()))
        
        return recon_loss, duration_loss, align_loss
        
    
    def inference(self, text, alpha=1.0, temperature=1.0):
        assert len(text)==1, 'You must encode only one sentence at once'
        text_lengths = torch.tensor([text.size(1)]).to(text.device)
        key, value = self.TextEnc(text)
        durations = self.get_duration(value)
        h, durations = self.LengthRegulator(value, durations, alpha)
        h = self.Compress(h).transpose(1,2)
        
        if isinstance(temperature, float):
            temperature=[temperature]*len(self.BVAE_blocks)
            
        for i, block in enumerate(reversed(self.BVAE_blocks)):
            h, _ = block.down(h, sample=True, temperature=temperature[i])

        mel_out = self.Projection(h)
        
        return mel_out, durations
    
    
    def get_duration(self, value, mask=None):
        durations = self.Duration(value.transpose(1,2).detach(), mask)
        return durations
    
    
    def align2duration(self, alignments, mel_lengths):
        max_ids = torch.max(alignments, dim=2)[1]
        max_ids_oh = F.one_hot(max_ids, alignments.size(2))
        mask = get_mask_from_lengths(mel_lengths//self.ratio).unsqueeze(-1)
        max_ids_oh.masked_fill_(mask, 0)
        durations = max_ids_oh.sum(dim=1).to(torch.float)
        return durations
    
    
    def LengthRegulator(self, hidden_states, durations, alpha=1.0):
        durations = torch.round(durations*alpha).to(torch.long)
        durations[durations<=0]=1
        return hidden_states.repeat_interleave(durations[0], dim=1), durations
    
    
    def jitter(self, alignments, mel_lengths):
        B, T, _ = alignments.size()
        batch_indices = torch.arange(B).unsqueeze(1).to(alignments.device)
        jitter_indices = torch.arange(T).unsqueeze(0).repeat(B,1).to(alignments.device)
        jitter_indices = torch.round(jitter_indices + (2*torch.rand(jitter_indices.size())-1).to(alignments.device)).to(torch.long)
        jitter_indices = torch.where(jitter_indices<(mel_lengths//self.ratio).unsqueeze(1),
                                     jitter_indices,
                                     ((mel_lengths//self.ratio)-1).unsqueeze(-1).repeat(1,T))
        jitter_indices[jitter_indices<=0]=0
        alignments = alignments[batch_indices, jitter_indices]
        alignments.masked_fill_(self.mel_mask[:,::self.ratio].unsqueeze(-1), 0)
        return alignments


