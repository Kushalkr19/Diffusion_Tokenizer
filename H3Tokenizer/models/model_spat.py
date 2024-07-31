import torch
import torch.nn as nn
from .models_mae_Spat import spat_mae_b
from .quantizers import VectorQuantizerLucid, Memcodes
import os

class SpatialModel(nn.Module):
    def __init__(self, config):
        super(SpatialOnlyModel, self).__init__()
        model_args = config['model']
        self.quant_args = config['quantizer']
        self.enc_dim = self.quant_args['latent_dim']
        self.norm_pix_loss = config['loss']['norm_pix_loss']
        
        # Initialize spatial encoder
        self.spatial_encoder = spat_mae_b(model_args, model_args['num_channels'])
        
        print('Loading pre-trained weights')
        self._load_weights_spat()
        
        # Set up quantizer
        self.quantizer = self._get_quantizer()
        
        self.debug = config['debug']
        
    def _load_weights_spat(self, weights_root: str="/work/tpanambur_umass_edu/Experiments/Algorithm/Moon/fdl-2024-lunar/H3Tokenizer/models/hypersigma_weights/", name: str="spat-vit-base-ultra-checkpoint-1599.pth"):
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
                dim=self.quant_args['latent_dim'],
                codebook_size=self.quant_args['codebook_size'],
                codebook_dim=self.enc_dim,
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

        spat_enc_reshaped = spat_enc.transpose(1, 2).reshape(-1, self.enc_dim, spat_Hp, spat_Wp)
        if self.debug: print("spat_enc_reshaped:", spat_enc_reshaped.shape)

        return spat_enc_reshaped, spat_ids_restore, spat_mask

    def decoder(self, quant_enc, spat_ids_restore):
        B, _, _, _ = quant_enc.shape
        spat_latent = quant_enc.reshape(B, self.enc_dim, -1).transpose(1, 2)
        if self.debug: print("spat_latent:", spat_latent.shape)
        
        spat_dec = self.spatial_encoder.forward_decoder(spat_latent, spat_ids_restore)
        if self.debug: print("spat_dec:", spat_dec.shape)
        
        spat_dec = self.spatial_encoder.unpatchify(spat_dec)
       
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
        spat_dec = spat_dec.detach().cpu()

        # Return the same structure as SpatSpecModel, with None for unused components
        return spat_recon_loss, None, code_loss, tokens, spat_dec, None
    
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