import torch
import torch.nn as nn
from .models_mae_Spat import spat_mae_b
from .quantizers import VectorQuantizerLucid, Memcodes
import os
from .percept_loss import TimmPerceptualLoss


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class ConvNeXtBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch.

    From https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0.,kernel_size=7, padding=3, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim) # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
    
class SpatialModel(nn.Module):
    def __init__(self, config):
        super(SpatialModel, self).__init__()
        model_args = config['model']
        self.quant_args = config['quantizer']
        self.enc_dim = config['model']['enc_dim']
        self.latent_dim = self.quant_args['latent_dim']
        self.norm_pix_loss = config['loss']['norm_pix_loss']
        self.perceptloss = None
        if config['loss']['use_percept']:
            self.perceptloss = TimmPerceptualLoss(config['loss']['model_id'],config['loss']['feature_ids']).eval()
            
        
        # Initialize spatial encoder
        self.spatial_encoder = spat_mae_b(model_args, model_args['num_channels'], drop_path_rate=model_args['drop_path_rate'])
        
        print('Loading pre-trained weights')
        self._load_weights_spat()
        
        # Set up quantizer
        self.quantizer = self._get_quantizer()
        if config['post_mlp']:
            print('MLP')
            self.norm_mlp_enc = nn.LayerNorm(model_args['enc_dim'])
            self.post_mlp_enc = Mlp(model_args['enc_dim'], int(config['mlp_ratio']*model_args['enc_dim']), act_layer=nn.Tanh)
            self.norm_mlp_dec = nn.LayerNorm(model_args['enc_dim'])
            self.post_mlp_dec = Mlp(model_args['enc_dim'], int(config['mlp_ratio']*model_args['enc_dim']), act_layer=nn.Tanh)
            
        if self.enc_dim!=self.latent_dim:
            print('Dim reduction')
            self.quant_proj = torch.nn.Conv2d(self.enc_dim, self.latent_dim, 1)
            self.post_quant_proj = torch.nn.Conv2d(self.latent_dim, self.enc_dim, 1)
            
        if config['model']['out_conv']:
            print('Convnext End', model_args['out_conv_type'])
            if model_args['out_conv_type']==1:
                self.out_conv = nn.Sequential(ConvNeXtBlock(dim=model_args['num_channels']),
                                              ConvNeXtBlock(dim=model_args['num_channels']))
            elif model_args['out_conv_type']==2:
                self.out_conv = nn.Sequential(ConvNeXtBlock(dim=model_args['num_channels']),
                                              ConvNeXtBlock(dim=model_args['num_channels']),
                                              ConvNeXtBlock(dim=model_args['num_channels']),
                                            ConvNeXtBlock(dim=model_args['num_channels']))
            elif model_args['out_conv_type']==3:
                self.out_conv = nn.Sequential(ConvNeXtBlock(dim=model_args['num_channels'],kernel_size=3,padding=1),
                                              ConvNeXtBlock(dim=model_args['num_channels'],kernel_size=3,padding=1),
                                            ConvNeXtBlock(dim=model_args['num_channels']),
                                            ConvNeXtBlock(dim=model_args['num_channels']))
                                                      
        self.debug = config['debug']
        
    def _load_weights_spat(self, weights_root: str="models/hypersigma_weights/", name: str="spat-vit-base-ultra-checkpoint-1599.pth"):
        if not os.path.exists(os.path.join(weights_root, name)):
            return 
        model = torch.load(os.path.join(weights_root, name), map_location=torch.device('cpu'))
        available_weight_keys = model['model'].keys()

        for name, param in self.spatial_encoder.named_parameters():
            if name in available_weight_keys:
                if param.data.shape == model['model'][name].shape:
                    param.data = model['model'][name]
                    
        print('=================Loaded Spat Weights==================')

    def _get_quantizer(self):
        if self.quant_args['quant_type'] == 'lucid':
            return VectorQuantizerLucid(
                dim=self.latent_dim,
                codebook_size=self.quant_args['codebook_size'],
                codebook_dim=self.latent_dim,
                heads=self.quant_args['num_codebooks'],
                use_cosine_sim=self.quant_args.get('norm_codes', False),
                threshold_ema_dead_code=self.quant_args.get('threshold_ema_dead_code', 2),
                code_replacement_policy=self.quant_args.get('code_replacement_policy', 'random'),
                sync_codebook=self.quant_args.get('sync_codebook', False),
                decay=self.quant_args.get('ema_decay', 0.99),
                commitment_weight=self.quant_args['commitment_weight'],
                norm_latents=self.quant_args.get('norm_latents', False),
                kmeans_init=self.quant_args.get('kmeans_init', False),
            )
        elif self.quant_args['quant_type'] == 'memcodes':
            return Memcodes(
                dim=self.quant_args['latent_dim'],
                codebook_size=self.quant_args['codebook_size'],
                heads=self.quant_args['num_codebooks'],
                temperature=1.0,
            )
        else:
            raise ValueError(f'{self.quant_args["quant_type"]} not a valid quant_type.')

    def encoder(self, x, mask):
        
        spat_enc, spat_mask, spat_ids_restore, (spat_Hp, spat_Wp) = self.spatial_encoder.forward_encoder(x, mask)
        if self.debug: print("spat_enc:", spat_enc.shape)
        
        if hasattr(self, 'post_mlp_enc'):
            spat_enc = spat_enc.float() + self.post_mlp_enc(self.norm_mlp_enc(spat_enc.float()))
            if self.debug: print("post_mlp_spat_enc:", spat_enc.shape)

        spat_enc = spat_enc.transpose(1, 2).reshape(-1, self.enc_dim, spat_Hp, spat_Wp)
        if self.debug: print("spat_enc_reshaped:", spat_enc.shape) 
        
        if hasattr(self, 'quant_proj'):
            spat_enc = self.quant_proj(spat_enc)
            if self.debug: print("spat_enc_quant_proj:", spat_enc.shape)

        return spat_enc, spat_ids_restore, spat_mask

    def decoder(self, quant_enc, spat_ids_restore):
        
        B, _, _, _ = quant_enc.shape
        if hasattr(self, 'post_quant_proj'):
            quant_enc = self.post_quant_proj(quant_enc)
            if self.debug: print("spat_dec_quant_proj:", quant_enc.shape)
            
        spat_latent = quant_enc.reshape(B, self.enc_dim, -1).transpose(1, 2)
        if self.debug: print("spat_latent:", spat_latent.shape)
        
        if hasattr(self, 'post_mlp_dec'):
            spat_latent = spat_latent.float() + self.post_mlp_dec(self.norm_mlp_dec(spat_latent.float()))
            if self.debug: print("post_mlp_spat_dec:", spat_latent.shape)

        
        spat_dec = self.spatial_encoder.forward_decoder(spat_latent, spat_ids_restore)
        if self.debug: print("spat_dec:", spat_dec.shape)
            
        
        spat_dec = self.spatial_encoder.unpatchify(spat_dec)
        
        
        if hasattr(self, 'out_conv'):
            spat_dec = self.out_conv(spat_dec)
            if self.debug: print("out_conv_spat_dec:", spat_dec.shape)
       
        return spat_dec

    def forward(self, x, nan_mask=None, enc_mask=0.0):
        if self.debug: print("x:", x.shape)
        if self.debug: print("nan_mask:", nan_mask.shape)
        
        if nan_mask is None:
            nan_mask = torch.ones_like(x, dtype=bool)
        
        # Encoder
        spat_proj, spat_ids_restore, spat_mask = self.encoder(x, enc_mask)

        # Quantization
        quant_enc, code_loss, tokens = self.quantizer(spat_proj)
        if self.debug: print("quant_enc:", quant_enc.shape)
        
        # Decoder
        spat_dec = self.decoder(quant_enc, spat_ids_restore)
        spat_recon_loss = self.recon_loss(x, spat_dec, nan_mask)
        with torch.no_grad():
            spat_percept_loss = self.perceptloss(x,spat_dec, True)
        spat_dec = spat_dec.detach().cpu()
        

        # Return the same structure as SpatSpecModel, with None for unused components
        return spat_recon_loss, None, spat_percept_loss, code_loss, tokens, spat_dec, None
    
    def recon_loss(self, input, target, nan_mask):
        nan_mask = ~nan_mask.bool()

        if self.norm_pix_loss:
            # Normalize target
            mean = torch.sum(target * nan_mask.unsqueeze(1), dim=(2, 3), keepdim=True) / torch.sum(nan_mask.unsqueeze(1), dim=(2, 3), keepdim=True)
            var = torch.sum((target - mean).pow(2) * nan_mask.unsqueeze(1), dim=(2, 3), keepdim=True) / torch.sum(nan_mask.unsqueeze(1), dim=(2, 3), keepdim=True)
            target = torch.where(nan_mask.unsqueeze(1), (target - mean) / (var + 1.e-6).sqrt(), target)

            # Normalize input
            mean_input = torch.sum(input * nan_mask.unsqueeze(1), dim=(2, 3), keepdim=True) / torch.sum(nan_mask.unsqueeze(1), dim=(2, 3), keepdim=True)
            var_input = torch.sum((input - mean_input).pow(2) * nan_mask.unsqueeze(1), dim=(2, 3), keepdim=True) / torch.sum(nan_mask.unsqueeze(1), dim=(2, 3), keepdim=True)
            input = torch.where(nan_mask.unsqueeze(1), (input - mean_input) / (var_input + 1.e-6).sqrt(), input)

        # Calculate loss only for non-nan values
        diff = (input - target) * nan_mask
        loss = torch.sum(diff.pow(2)) / torch.sum(nan_mask)

        return loss

    
    def tokenize(self, x, mask=0.0):
        # Encoder
        spat_proj, _, _ = self.encoder(x, mask)

        # Quantization
        _, _, tokens = self.quantizer(spat_proj)
        return tokens