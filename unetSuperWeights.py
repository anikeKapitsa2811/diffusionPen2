from abc import abstractmethod
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
from inspect import isfunction
import math
import torchvision.models as models
import random
from transformers import CanineModel

def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)



class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        
        ctx.input_tensors = [x.float().detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor



def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding



def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)



class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        
        k = self.to_k(context)
        v = self.to_v(context)

        mask = None #torch.ones(1, 8192).bool().cuda('cuda:6')
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)
        


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    #'seq shape', seq.shape)
    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")



class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        #self.weights = ResNet18_Weights.DEFAULT
        
        #num_ftrs = self.image_encoder.fc.in_features
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention for the image
        #self.attnc = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention for the context
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)
    
    def _forward(self, x, context=None):
        
        x = self.attn1(self.norm1(x)) + x
        #print('x shape', x.shape)
        #print('context shape', context.shape)
        x = self.attn2(self.norm2(x), context=context, mask=None) + x
        x = self.ff(self.norm3(x)) + x
        return x


class Style_Text_Encoder(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        #self.weights = ResNet18_Weights.DEFAULT
        
        #num_ftrs = self.image_encoder.fc.in_features
        #self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention for the image
        #self.attnc = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention for the context
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)
    
    def _forward(self, x, context=None):
        
        #x = self.attn1(self.norm1(x)) + x
        x = self.attn2(x, context=context, mask=None) + x
        #x = self.attn2(self.norm2(x), context=context, mask=None) + x
        x = self.ff(self.norm3(x)) + x
        return x



class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None, part='encoder', vocab_size=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))
        self.part = part
    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        #print('x spatial trans in', x.shape)
        
        
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        if self.part != 'sca':
            x = rearrange(x, 'b c h w -> b (h w) c')
    
        for block in self.transformer_blocks:
            x = block(x, context=context)
        if self.part != 'sca':
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in



# dummy replace
def convert_module_to_f16(x):
    pass

def convert_module_to_f32(x):
    pass

def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb, context):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
                
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
                
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class TransposedUpsample(nn.Module):
    'Learned 2x upsampling without padding'
    def __init__(self, channels, out_channels=None, ks=5):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.up = nn.ConvTranspose2d(self.channels,self.out_channels,kernel_size=ks,stride=2)

    def forward(self,x):
        return self.up(x)


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = nn.Conv2d(#dims,
                 self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = nn.AvgPool2d(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(
                channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )


    def _forward(self, x, emb):
        
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
            
        # if context is None:
        #     context= torch.zeros(emb.shape).to(emb.device)
        
        # emb = torch.cat([emb, context], dim=-1)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        
        return self.skip_connection(x) + h



class Res_Block(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(
                channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )


    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h



class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(nn.Conv2d(channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)   # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        #return pt_checkpoint(self._forward, x)  # pytorch

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


##################################################################################

    
class Word_Attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Word_Attention, self).__init__()
        self.linear_query = nn.Linear(input_size, hidden_size)
        self.linear_key = nn.Linear(input_size, hidden_size)
        self.linear_value = nn.Linear(input_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        query = self.linear_query(x)
        key = self.linear_key(x)
        value = self.linear_value(x)
        
        # Calculate attention scores
        scores = query @ key.transpose(-2, -1)
        scores = self.softmax(scores)
        
        # Calculate weighted sum of the values
        word_embedding = scores @ value
        #print('word emb', word_embedding.shape)
        return word_embedding


class CharacterEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, max_seq_len):
        super(CharacterEncoder, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.attention = Word_Attention(hidden_size, hidden_size)

        self.embedding_dim = hidden_size
        self.max_seq_len = max_seq_len
        self.positional_encoding = self.get_positional_encoding()

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        #print('x before embedding', x.shape)
        x = self.embedding(x)
        #print('x', x.shape)
        x += self.positional_encoding[:x.size(1), :].to(x.device)
        word_embedding = x #self.attention(x)
        return word_embedding
    
    def get_positional_encoding(self):
        positional_encoding = torch.zeros(self.max_seq_len, self.embedding_dim)
        #print('pos enc', positional_encoding.shape)
        for pos in range(self.max_seq_len):
            for i in range(0, self.embedding_dim, 2):
                positional_encoding[pos, i] = math.sin(pos / (10000 ** (i / self.embedding_dim)))
                positional_encoding[pos, i + 1] = math.cos(pos / (10000 ** ((i + 1) / self.embedding_dim)))
        return positional_encoding

##################################################################################
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        # implementation details here
        super(ResNet, self).__init__()
    def forward(self, x):
        # forward pass implementation here
        
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        x = self.layer1(x)

        feat = self.layer2[0].conv1(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        feat = self.fc(feat)

        return feat



##################################################################################

class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=True,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=768,                 # custom transformer support
        vocab_size=80,                  # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=False,
        text_encoder=None,
        args=None, 
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            self.cont_dim = context_dim
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        self.args = args

        #if clip is not None:
            #self.clip = clip
            #print('clip', self.clip)
            #self.text_encoder = self.clip.text_encoder
            #self.tokenizer = self.clip.tokenizer

        self.text_encoder = text_encoder
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        #==================== INPUT BLOCK ====================
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

        #==================== MIDDLE BLOCK ====================
        
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        
        #==================== OUTPUT BLOCK ====================
        
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
            normalization(ch),
            nn.Conv2d(model_channels, n_embed, 1),
            nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
        )
        
        self.interpolation = args.interpolation
        self.mix_rate = args.mix_rate
        #self.style_lin = nn.Linear(1280*5, time_embed_dim)
        self.style_lin = nn.Linear(1280, time_embed_dim)
        self.target_token_idx = 0
        self.text_lin = nn.Linear(768, 320)
    
    
    def identify_super_weights_all(self, images, timesteps=None, context=None, style_features=None,
                                 device=None,top_k_per_module=5, num_runs=3, save_path=None,
                                scale_range=(0.8, 1.4)):
        """
        Inference-time detector of important parameters across UNet submodules.

        - Registers light forward hooks on Conv2d/Linear modules.
        - Runs `num_runs` forward passes with provided inputs.
        - Scores params by (module_activation_mean * mean_abs_weight).
        - Selects top_k_per_module params per module, maps scores -> gentle scales.
        - Returns {"by_module": {...}, "flat": {...}} and saves flat JSON if save_path set.

        Args:
        images (torch.Tensor): batch already (or will be) moved to device.
        timesteps (torch.LongTensor or None): (B,) timesteps or None -> zeros.
        context: tokenized text/context or None.
        style_features: style features tensor or None.
        top_k_per_module: top-K params to pick per module (<=0 => select all).
        num_runs: number of forward passes to aggregate.
        device: device string.
        save_path: optional path to save flat mapping JSON.
        scale_range: (lo, hi) multiplicative scale applied to selected params.

        Returns:
        dict: {"by_module": {module_name: {param_name: scale, ...}, ...},
                "flat": {param_name: scale, ...}}
        """
        import torch, json, torch.nn as nn

        # Move inputs to device
        images = images.to(device)
        if timesteps is None:
            timesteps = torch.zeros(images.size(0), dtype=torch.long, device=device)
        else:
            timesteps = timesteps.to(device)

        # Collect candidate modules (Conv2d and Linear)
        module_map = {}
        activation_stats = {}
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module_map[name] = module
                activation_stats[name] = {'sum': 0.0, 'count': 0}

        if len(module_map) == 0:
            return {"by_module": {}, "flat": {}}

        # Hook factory: store mean(|activation|) scalar per module
        hooks = []
        def make_hook(mname):
            def hook(mod, inp, out):
                val = out
                if isinstance(out, (tuple, list)):
                    val = out[0]
                if not torch.is_tensor(val):
                    return
                try:
                    scalar = float(val.detach().abs().mean().item())
                except Exception:
                    return
                activation_stats[mname]['sum'] += scalar
                activation_stats[mname]['count'] += 1
            return hook

        # Register hooks
        for mname, mod in module_map.items():
            try:
                hooks.append(mod.register_forward_hook(make_hook(mname)))
            except Exception:
                pass

        # Run forwards to collect activations (no grad)
        self.eval()
        with torch.no_grad():
            for _ in range(max(1, int(num_runs))):
                # Match the forward signature you use in sampling.
                # If your forward uses different arg names, adjust below accordingly.
                _ = self(images, timesteps=timesteps, context=context, y=None, style_extractor=style_features)

        # Remove hooks
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass

        # Compute per-module activation mean
        module_act_mean = {}
        for mname, stats in activation_stats.items():
            module_act_mean[mname] = (stats['sum'] / stats['count']) if stats['count'] > 0 else 0.0

        # Prepare param -> score mapping
        param_scores = {}  # pname -> (score, module_name)
        for pname, p in self.named_parameters():
            # Find longest module name that is prefix of pname
            best_mod = None
            best_len = -1
            for mname in module_act_mean.keys():
                prefix = mname + '.'
                if pname.startswith(prefix) and len(mname) > best_len:
                    best_mod = mname
                    best_len = len(mname)
            # fallback to parent path
            if best_mod is None:
                cand = pname.rpartition('.')[0]
                if cand in module_act_mean:
                    best_mod = cand
            w_mean = float(p.detach().abs().mean().item())
            act_mean = module_act_mean.get(best_mod, 0.0)
            score = act_mean * (w_mean + 1e-12)
            param_scores[pname] = (score, best_mod if best_mod is not None else "")

        # Group and pick top_k_per_module
        by_module = {}
        flat_mapping = {}
        lo, hi = float(scale_range[0]), float(scale_range[1])
        for mname in module_act_mean.keys():
            items = [(pname, val[0]) for pname, val in param_scores.items() if val[1] == mname]
            if not items:
                continue
            items.sort(key=lambda x: x[1], reverse=True)
            k = top_k_per_module if (top_k_per_module and top_k_per_module > 0) else len(items)
            sel = items[:min(k, len(items))]
            scores = [s for _, s in sel]
            smin, smax = (min(scores), max(scores)) if scores else (0.0, 0.0)
            module_map_out = {}
            for pname, score in sel:
                norm = (score - smin) / (smax - smin) if smax > smin else 0.5
                scale = float(lo + norm * (hi - lo))
                module_map_out[pname] = scale
                flat_mapping[pname] = scale
            by_module[mname] = module_map_out

        # Optionally save flat mapping
        if save_path:
            try:
                with open(save_path, 'w') as f:
                    json.dump(flat_mapping, f, indent=2)
            except Exception:
                pass

        return {"by_module": by_module, "flat": flat_mapping}


 
    import json

    def identify_super_weights_per_weight(
        self,
        images,
        timesteps=None,
        context=None,
        style_features=None,
        top_k_per_module=5,
        num_runs=3,
        device=None,
        save_path=None,
    ):
        """
        Identify individual weight importances for all connections in the U-Net model (diffusion U-Net).

        Args:
            images (torch.Tensor): Input batch of images (real or synthetic) (B, C, H, W).
            timesteps (torch.LongTensor): Diffusion timesteps (can be zero if not provided).
            context (torch.Tensor or None): Tokenized text or conditioning for context (optional).
            style_features (torch.Tensor or None): Style conditioning features (optional, can be None).
            top_k_per_module (int): Number of top important weights to keep per module (or all if <= 0).
            num_runs (int): Forward passes to aggregate importance scores.
            device (str): Compute device (e.g., "cuda" or "cpu").
            save_path (str or None): Path to save the results (optional, saves as JSON).

        Returns:
            dict: Top-k individual weight importance scores per parameter.
                Format: { "param_name": [ {"index": <tuple>, "importance": <float>} ] }
        """
        # Move model and inputs to the desired device
        self.to(device)
        images = images.to(device)
        if timesteps is not None:
            timesteps = timesteps.to(device)
        else:
            # Default to zero timesteps if not provided
            timesteps = torch.zeros(images.size(0), dtype=torch.long, device=device)
        if context is not None:
            context = context.to(device)
            
        style_extractor=style_features
        if style_features is not None:
            style_features = style_extractor.to(device)
        # check size of each 
        
        print('2.images', images.shape)
        print("2.timesteps", timesteps.shape)
        #print("2.context.shape:",context.shape)
        print("2.style_features:", style_extractor.shape if style_extractor is not None else None)
        self.eval()

        # Initialize storage for weight importance scores
        weight_importance = {
            name: torch.zeros_like(p, device=device) for name, p in self.named_parameters()
        }

        for _ in range(num_runs):
            self.zero_grad(set_to_none=True)

            # Forward pass with additional inputs
            out = self(images, timesteps=timesteps, context=context,y=None,mix_rate=None, style_extractor=style_extractor)
            objective = out.abs().mean()  # Scalar objective to backprop importance
            objective.backward()

            # Accumulate per-weight gradient × weight scores
            for name, p in self.named_parameters():
                if p.grad is not None:
                    weight_importance[name] += (p.grad * p).abs()

        # Normalize scores by the number of runs
        for name in weight_importance:
            weight_importance[name] = weight_importance[name].div_(num_runs).detach().cpu()

        # Select top-k important weights per module if specified
        top_k_weights = {}
        for name, scores in weight_importance.items():
            flat_scores = scores.flatten()  # Flatten the weights
            k = min(top_k_per_module, flat_scores.numel()) if top_k_per_module > 0 else flat_scores.numel()
            top_vals, top_idxs = torch.topk(flat_scores, k)

            # Map the top scores to their original indices
            top_weights = [
                {"index": np.unravel_index(idx.item(), scores.shape), "importance": val.item()}
                for idx, val in zip(top_idxs, top_vals)
            ]
            top_k_weights[name] = top_weights

        # Save results as JSON if save_path is specified
        if 0:#save_path:
            with open(save_path, "w") as f:
                json.dump(top_k_weights, f, indent=2)

        return top_k_weights

    

    
    #url=https://github.com/anikeKapitsa2811/DiffusionPen/blob/main/unet.py
    def identify_super_weights(self, images, timesteps=None, context=None, style_features=None,
                            top_k=50, num_runs=3, device='cuda', save_path=None):
        """
        Simple inference-time superweight detector.

        Args:
        images (torch.Tensor): a batch of images (B,C,H,W) on CPU/GPU.
        timesteps (torch.LongTensor or None): (B,) timesteps; if None uses zeros.
        context: tokenized text/context as used by your model (or None).
        style_features: style tensor or None.
        top_k: number of parameter entries to return.
        num_runs: how many forward runs to aggregate activations (small e.g. 1-5).
        device: device string, e.g. args.device.
        save_path: optional path to save JSON mapping {param_name: scale}.

        Returns:
        dict: {state_dict_param_name: scale (float)} for the top_k parameters.
        """
        images = images.to(device)
        if timesteps is None:
            timesteps = torch.zeros(images.size(0), dtype=torch.long, device=device)

        # Collect modules of interest
        activation_stats = {}        # module -> {'sum': float, 'count': int}
        module_name_of = {}         # module -> name
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module_name_of[module] = name
                activation_stats[module] = {'sum': 0.0, 'count': 0}

        # register hooks
        hooks = []
        def make_hook(mod):
            def hook(mod, inp, out):
                # scalar aggregate of absolute activation
                val = float(out.detach().abs().mean().item())
                activation_stats[mod]['sum'] += val
                activation_stats[mod]['count'] += 1
            return hook

        for mod in list(module_name_of.keys()):
            hooks.append(mod.register_forward_hook(make_hook(mod)))

        # run a few forward passes (inference)
        self.eval()
        with torch.no_grad():
            for _ in range(max(1, int(num_runs))):
                # forward signature matches how you call the model in the repo:
                _ = self(images, timesteps=timesteps, context=context, y=None, style_extractor=style_features)

        # remove hooks
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass

        # map module activations -> parameter scores
        param_scores = {}
        # Precompute param id -> (name, tensor) for the whole model
        all_params = {id(p): (n, p) for n, p in self.named_parameters()}

        for mod, stats in activation_stats.items():
            if stats['count'] == 0:
                continue
            act_mean = stats['sum'] / stats['count']
            # collect params that belong to this module by object identity
            for p in mod.parameters():
                entry = all_params.get(id(p), None)
                if entry is None:
                    continue
                pname, ptensor = entry
                w_mean = float(ptensor.detach().abs().mean().item())
                score = act_mean * (w_mean + 1e-12)
                # if multiple module hooks touch same param, keep the max score
                if pname in param_scores:
                    param_scores[pname] = max(param_scores[pname], score)
                else:
                    param_scores[pname] = score

        if len(param_scores) == 0:
            # nothing found
            if save_path:
                with open(save_path, 'w') as f:
                    json.dump({}, f, indent=2)
            return {}

        # select top_k parameters by score
        sorted_items = sorted(param_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_items = sorted_items[:min(top_k, len(sorted_items))]
        scores = [s for _, s in top_k_items]
        smin, smax = min(scores), max(scores)

        # normalize to a gentle multiplicative scale range (example: 0.8..1.4)
        lo, hi = 0.8, 1.4
        mapping = {}
        for name, score in top_k_items:
            if smax > smin:
                norm = (score - smin) / (smax - smin)
            else:
                norm = 0.5
            scale = float(lo + norm * (hi - lo))
            mapping[name] = scale

        # save if requested
        if save_path:
            try:
                with open(save_path, 'w') as f:
                    json.dump(mapping, f, indent=2)
            except Exception:
                pass

        return mapping
    
    
    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)
  
    def forward(self, x, timesteps=None, context=None, y=None, mix_rate=None, style_extractor=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        #print('y', y.shape)
        
        # assert (y is not None) == (
        #     self.num_classes is not None
        # ), "must specify y if and only if the model is class-conditional"
        hs = []
        
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)#.to(x.device)
        t_emb = t_emb.to(x.device)
        print('t_emb', t_emb.device," x.device:",x.device)
        emb = self.time_embed(t_emb)
        
        
        #if self.num_classes is not None:
         #   assert y.shape == (x.shape[0],)
        if style_extractor is not None:
            s_id = style_extractor
            y = s_id.to(x.device)
           
        if self.interpolation:
            
            s1 = random.randint(0, 338)
            s2 = random.randint(0, 338)
            while s1 == s2:
                s2 = random.randint(0, 338)
            y1 = torch.tensor([s1]).long().to(x.device)
            y2 = torch.tensor([s2]).long().to(x.device)
            y1 = self.label_emb(y1).to(x.device)
            y2 = self.label_emb(y2).to(x.device)
            y = (1-self.mix_rate)*y1 + self.mix_rate*y2
            
            y = y.to(x.device)
            emb = emb + y  
        else:
            if style_extractor is not None:
                
                b, e = emb.shape
                
                y = y.reshape(b, 5, -1)
                y = torch.mean(y, dim=1)

                noise=False
                if noise==True:
                    magn = torch.norm(y, dim=1, keepdim=True)
                    noise = torch.randn_like(y)*0.25
                    #bernoulli mask in noise
                    noise = noise*torch.bernoulli(torch.ones_like(noise)*0.2)
                    
                    y = y + noise
                    y = magn * y / torch.norm(y, dim=1, keepdim=True)
                
                y = self.style_lin(y)
                
                emb = emb + y 
              
            else:
                emb = emb + self.label_emb(y)
            
        if context is not None:
            
            context = self.text_encoder(**context).last_hidden_state#.to(x.device)
           
            if self.cont_dim == 320:
                context = self.text_lin(context)#.unsqueeze(1)
                
        h = x.type(self.dtype)
        context = context.to(h.device)
        
        #INPUT BLOCKS
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        
        #MIDDLE BLOCK
        h = self.middle_block(h, emb, context)
        
        #OUTPUT BLOCKS
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
            
        h = h.type(x.dtype)
        
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            
            return self.out(h)




