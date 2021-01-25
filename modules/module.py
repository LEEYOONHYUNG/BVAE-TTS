import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as sn
import torch.distributions as D
from .conv import *


class Softplus(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result=torch.log(1+torch.exp(i))
        ctx.save_for_backward(i)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output*torch.sigmoid(ctx.saved_variables[0])


class CustomSoftplus(nn.Module):
    def forward(self, input_tensor):
        return Softplus.apply(input_tensor)



class TextEnc(nn.Module):
    def __init__(self, hp):
        super(TextEnc, self).__init__()
        self.Embedding = nn.Embedding(hp.n_symbols, hp.hidden_dim)
        self.conv_layers = nn.ModuleList([Conv1d(hp.hidden_dim, 2*hp.hidden_dim, hp.kernel_size) for _ in range(7)])
    
    def forward(self, text, mask=None):
        embedded = F.dropout(self.Embedding(text), 0.1, training=self.training)
        x = embedded.transpose(1,2)
        
        for conv in self.conv_layers:
            x1, x2 = torch.chunk( conv(x, mask), 2, dim=1)
            x = (x1 * torch.sigmoid(x2) + x) / 2**0.5
            x = F.dropout(x, 0.1, training=self.training)
            
        key = x.transpose(1, 2)
        value = (key+embedded)/2**0.5
        return key, value



class BVAE_block(nn.Module):
    def __init__(self, hdim, kernel_size, n_layers, down_upsample):
        super(BVAE_block, self).__init__()
        self.down_upsample=down_upsample
        self.BVAE_layers = nn.ModuleList()
        for i in range(n_layers):
            self.BVAE_layers.append(BVAE_layer(hdim,
                                               kernel_size,
                                               dilation=2**i,
                                               adj_dim=( (down_upsample=='F') and (i==0) )))
        
    def up(self, inputs, mask=None):
        if self.down_upsample=='T':
            inputs = self.blur_pool(inputs, mask)
        x = inputs
        for layer in self.BVAE_layers:
            x = layer.up(x, mask)

        return x
    
    
    def down(self, inputs, mask=None, sample=False, temperature=1.0):
        x = inputs
        kl=0
        for layer in reversed(self.BVAE_layers):
            x, curr_kl = layer.down(x, mask, sample, temperature) 
            kl += curr_kl
        
        if self.down_upsample=='T':
            x = x.repeat_interleave(2,-1)
        
        return x, kl
    
    
    def blur_pool(self, x, mask):
        blur_kernel = (torch.tensor([[[0.25,0.5,0.25]]])).repeat(x.size(1),1,1).to(x.device)
        outputs = F.conv1d(x, blur_kernel, padding=1, stride=2, groups=x.size(1))
        outputs = outputs.masked_fill(mask.unsqueeze(1), 0)
        return outputs
    
    
    
class BVAE_layer(nn.Module):
    def __init__(self, hdim, kernel_size, dilation=1, adj_dim=False):
        super(BVAE_layer, self).__init__()
        self.softplus = CustomSoftplus()
        
        ####################### BOTTOM_UP #########################
        if adj_dim==True:
            self.pre_conv = Conv1d(2*hdim, hdim, kernel_size, activation=F.elu, dilation=dilation)
        else:
            self.pre_conv = Conv1d(hdim, hdim, kernel_size, activation=F.elu, dilation=dilation)
            
        self.up_conv_a = nn.ModuleList([sn(Conv1d(hdim, hdim, kernel_size, activation=F.elu)),
                                        sn(Conv1d(hdim, 3*hdim, kernel_size, bias=False))])
        self.up_conv_b = sn(Conv1d(hdim, hdim, kernel_size, activation=F.elu))
        
        ######################## TOP_DOWN ##########################
        self.down_conv_a = nn.ModuleList([sn(Conv1d(hdim, hdim, kernel_size, activation=F.elu)),
                                          sn(Conv1d(hdim, 5*hdim, kernel_size, bias=False))])
        self.down_conv_b = nn.ModuleList([sn(Conv1d(2*hdim, hdim, kernel_size, bias=False)),
                                          sn(Conv1d(hdim, hdim, kernel_size, activation=F.elu))])
        
        if adj_dim==True:
            self.post_conv = Conv1d(hdim, 2*hdim, kernel_size, activation=F.elu, dilation=dilation)
        else:
            self.post_conv = Conv1d(hdim, hdim, kernel_size, activation=F.elu, dilation=dilation)
        
        
    def up(self, inputs, mask=None):
        inputs = self.pre_conv(inputs, mask)
        x = self.up_conv_a[0](inputs, mask)
        self.qz_mean, self.qz_std, h = self.up_conv_a[1](x, mask).chunk(3, 1)
        self.qz_std = self.softplus(self.qz_std)
        h = self.up_conv_b(h, mask)

        return (inputs+h)/2**0.5
    
    
    def down(self, inputs, mask=None, sample=False, temp=1):
        x = self.down_conv_a[0](inputs, mask)
        pz_mean, pz_std, rz_mean, rz_std, h = self.down_conv_a[1](x, mask).chunk(5, 1)
        pz_std, rz_std = self.softplus(pz_std), self.softplus(rz_std)
        
        if sample==True:
            prior = D.Normal(pz_mean, pz_std*temp)
            z = prior.rsample()
            kl = torch.zeros(inputs.size(0)).to(inputs.device).mean()
            
        else:
            prior = D.Normal(pz_mean, pz_std)
            posterior = D.Normal(pz_mean+self.qz_mean+rz_mean, pz_std*self.qz_std*rz_std)
            z = posterior.rsample().masked_fill(mask.unsqueeze(1), 0)
            kl = D.kl.kl_divergence(posterior, prior).mean()
            
        h = torch.cat((z, h), 1)
        h = self.down_conv_b[0](h, mask)
        h = self.down_conv_b[1](h, mask)
        outputs = self.post_conv((inputs+h)/2**0.5, mask)
        
        return outputs, kl
    

class DurationPredictor(nn.Module):
    def __init__(self, hp):
        super(DurationPredictor, self).__init__()
        self.conv1 = Conv1d(hp.hidden_dim, hp.hidden_dim, 3, bias=False, activation=F.elu)
        self.conv2 = Conv1d(hp.hidden_dim, hp.hidden_dim, 3, bias=False, activation=F.elu)
        
        self.ln1 = nn.LayerNorm(hp.hidden_dim)
        self.ln2 = nn.LayerNorm(hp.hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
        self.linear = Linear(hp.hidden_dim, 1)

    def forward(self, h, mask=None):
        x = self.conv1(h, mask)
        x = self.dropout(self.ln1(x.transpose(1,2)))
        x = self.conv2(x.transpose(1,2), mask)
        x = self.dropout(self.ln2(x.transpose(1,2)))
        out = self.linear(x).exp()+1
        
        return out.squeeze(-1)
    


class Prenet(nn.Module):
    def __init__(self, hp):
        super(Prenet, self).__init__()
        self.layers = nn.ModuleList([Conv1d(hp.n_mel_channels, hp.hidden_dim, 1, bias=True, activation=F.elu),
                                     Conv1d(hp.hidden_dim, hp.hidden_dim, 1, bias=True, activation=F.elu)])

    def forward(self, x, mask=None):
        for i, layer in enumerate(self.layers):
            x = F.dropout(layer(x, mask), 0.5, training=True)
        return x



class Projection(nn.Module):
    def __init__(self, hdim, kernel_size, outdim):
        super(Projection, self).__init__()
        self.layers=nn.ModuleList([Conv1d(hdim, hdim, kernel_size, activation=F.elu),
                                   Conv1d(hdim, hdim, kernel_size, activation=F.elu),
                                   Conv1d(hdim, outdim, 5)])
        
    def forward(self, x, mask=None):
        for i, layer in enumerate(self.layers):
            if i<len(self.layers)-1:
                x = F.dropout(layer(x, mask), 0.5, training=self.training)
            else:
                x = layer(x, mask)
        return torch.sigmoid(x)

