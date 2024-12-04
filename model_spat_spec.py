import torch
import torch.nn as nn
from .models_mae_Spat import spat_mae_b
from .models_mae_Spec import spec_mae_b
from .ml_4m.fourm.vq.quantizers import VectorQuantizerLucid, Memcodes
import os

class SpatSpecModel(nn.Module):
    def __init__(self, config):
        super(SpatSpecModel, self).__init__()
        model_args = config['model']
        self.quant_args = config['quantizer']
        self.enc_dim = self.quant_args['latent_dim']
        self.num_spec_tokens = model_args['num_tokens']
        self.norm_pix_loss = config['loss']['norm_pix_loss']
        
        # Initialize spatial and spectral encoders
        self.spatial_encoder = spat_mae_b(model_args, model_args['num_channels'])
        self.spectral_encoder = spec_mae_b(model_args, model_args['num_channels'])
        
        print('Loading pre-trained weights')
        #Load Pretrained
        self._load_weights_spat()
        self._load_weights_spec()
          
        # Set up quantizer
        self.quantizer = self._get_quantizer()
        
        # Calculate sizes for linear layers
        self.spat_size = model_args['image_size'] // model_args['patch_size']  # Assuming square layout
        self.spec_size = int(self.num_spec_tokens**0.5)  # Assuming square layout
        
        # Define linear layers for upsampling and downsampling
        self.spec_to_spat_proj = nn.Linear(self.num_spec_tokens, self.spat_size**2)
        self.fusion_conv = nn.Conv2d(self.enc_dim * 2, self.enc_dim, 1)
        self.spat_to_spec_proj = nn.Linear(self.spat_size**2, self.num_spec_tokens)
        self.debug = config['debug']
        self.spat_spec = config['model']['spat_spec']
        
    def _load_weights_spat(self, weights_root: str="models/pretrained/hypersigma_weights/", name: str="spat-vit-base-ultra-checkpoint-1599.pth"):
        """Loads matching weights into the SPATIAL model.

        Args:
            weights_root (str, optional): Path to where the weights are stored. Defaults to "H3Tokenizer/models/hypersigma_weights/".
            name (str, optional): Filename of weights to load. Defaults to "spat-vit-base-ultra-checkpoint-1599.pth".
        """
        if not os.path.exists(os.path.join(weights_root, name)):
            return 
        model = torch.load(os.path.join(weights_root, name), map_location=torch.device('cpu'))
        available_weight_keys = model['model'].keys()

        for name, param in self.spatial_encoder.named_parameters():
            if name in available_weight_keys:
                if param.data.shape == model['model'][name].shape:
                    param.data = model['model'][name]
                    
        print('=================Loaded Spat Weights==================')

    def _load_weights_spec(self, weights_root: str="models/pretrained/hypersigma_weights/", name: str="spec-vit-base-ultra-checkpoint-1599.pth"):
        """Loads matching weights into the SPECTRAL model.

        Args:
            weights_root (str, optional): Path to where the weights are stored. Defaults to "H3Tokenizer/models/hypersigma_weights/".
            name (str, optional): Filename of weights to load. Defaults to "spec-vit-base-ultra-checkpoint-1599.pth".
        """
        if not os.path.exists(os.path.join(weights_root, name)):
            return 
        model = torch.load(os.path.join(weights_root, name), map_location=torch.device('cpu'))
        available_weight_keys = model['model'].keys()

        for name, param in self.spectral_encoder.named_parameters():
            if name in available_weight_keys:
                if param.data.shape == model['model'][name].shape:
                    param.data = model['model'][name]
                    
        print('=================Loaded Spec Weights==================')

    def _get_quantizer(self):
        # Get quantizer based on config
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
        # Spatial encoding
        B,_,_,_ = x.shape
        spat_enc, spat_mask, spat_ids_restore, (spat_Hp, spat_Wp) = self.spatial_encoder.forward_encoder(x, mask)
        if self.debug: print("spat_enc:", spat_enc.shape)  # (B, N, D) = (B, 256, 768)

        # Spectral encoding
        spec_enc, spec_mask, spec_ids_restore, _ = self.spectral_encoder.forward_encoder(x, mask)
        if self.debug: print("spec_enc:", spec_enc.shape)  # (B, N, D) = (B, 36, 768)

        # Reshape and transpose spatial encoding
        spat_enc_reshaped = spat_enc.transpose(1, 2).reshape(-1, self.enc_dim, spat_Hp, spat_Wp)
        if self.debug: print("spat_enc_reshaped:", spat_enc_reshaped.shape)  # (B, 768, 16, 16)

        # Project spectral encoding to match spatial dimensions
        spec_enc_proj = self.spec_to_spat_proj(spec_enc.transpose(1, 2))
        if self.debug: print("spec_enc_proj:", spec_enc_proj.shape)
        spec_enc_reshaped = spec_enc_proj.reshape(-1, self.enc_dim, spat_Hp, spat_Wp)
        if self.debug: print("spec_enc_reshaped:", spec_enc_reshaped.shape)  # (B, 768, 16, 16)

        # Concatenate spatial and spectral encoding
        fused_enc = torch.cat([spat_enc_reshaped, spec_enc_reshaped], dim=1)
        if self.debug: print("fused_enc:", fused_enc.shape)  # (B, 1536, 16, 16)

        # Fusion convolution
        fused_proj = self.fusion_conv(fused_enc)
        if self.debug: print("fused_proj:", fused_proj.shape)  # (B, 768, 16, 16)

        return fused_proj, spat_ids_restore, spec_ids_restore, spat_mask, spec_mask

    def spat_spec_decoder(self, quant_enc, spat_ids_restore, spec_ids_restore, spec_mask):
        # Spatial decoding
        B,_,_,_ = quant_enc.shape
        spat_latent = quant_enc.reshape(B, self.enc_dim, -1).transpose(1, 2)
        if self.debug: print("spat_latent:", spat_latent.shape)
        spat_dec = self.spatial_encoder.forward_decoder(spat_latent, spat_ids_restore)
        if self.debug: print("spat_dec:", spat_dec.shape)  # (B, 72, 128, 128)

        # Project quantized encoding back to spectral dimensions
        quant_enc_flat = quant_enc.reshape(quant_enc.size(0), self.enc_dim, -1)
        if self.debug: print("quant_enc_flat:", quant_enc_flat.shape)
        spec_latent_proj = self.spat_to_spec_proj(quant_enc_flat).transpose(1, 2)
        if self.debug: print("spec_latent_proj:", spec_latent_proj.shape)  # (B, 36, 768)

        # Spectral decoding
        spec_dec, _ = self.spectral_encoder.forward_decoder(spec_latent_proj, spec_ids_restore, spec_mask)
        if self.debug: print("spec_dec:", spec_dec.shape)  # (B, 72, 128, 128)
        
        spat_dec = self.spatial_encoder.unpatchify(spat_dec)
        spec_dec = self.spectral_encoder.unpatchify(spec_dec)

        return spat_dec, spec_dec
    
    
    def spat_decoder(self, quant_enc, spat_ids_restore, spec_ids_restore, spec_mask):
        # Spatial decoding
        B,_,_,_ = quant_enc.shape
        spat_latent = quant_enc.reshape(B, self.enc_dim, -1).transpose(1, 2)
        if self.debug: print("spat_latent:", spat_latent.shape)
        
        spat_dec = self.spatial_encoder.forward_decoder(spat_latent, spat_ids_restore)
        if self.debug: print("spat_dec:", spat_dec.shape)  # (B, 72, 128, 128)
        
        spat_dec = self.spatial_encoder.unpatchify(spat_dec)
       
        return spat_dec, None

    def forward(self, x, nan_mask=None, enc_mask=0.0):
        
        if self.debug: print("x:", x.shape)
        if self.debug: print("nan_mask:", nan_mask.shape)
        
        if nan_mask is None:
            nan_mask = torch.ones_like(x, dtype=bool)
        
        #Encoder
        fused_proj, spat_ids_restore, spec_ids_restore, spec_mask, _ = self.encoder(x, enc_mask)

        # Quantization
        quant_enc, code_loss, tokens = self.quantizer(fused_proj)
        if self.debug: print("quant_enc:", quant_enc.shape)  # (B, 768, 16, 16)
        
        #Decoder
        if self.spat_spec:
            spat_dec, spec_dec = self.spat_spec_decoder(quant_enc, spat_ids_restore, spec_ids_restore, spec_mask)
            spat_recon_loss = self.recon_loss(x, spat_dec, nan_mask)
            spec_recon_loss = self.recon_loss(x, spec_dec, nan_mask)
            spat_dec = spat_dec.detach().cpu()
            spec_dec = spec_dec.detach().cpu()
            
        else:
            spat_dec, spec_dec = self.spat_decoder(quant_enc, spat_ids_restore, spec_ids_restore, spec_mask)
            spat_recon_loss = self.recon_loss(x, spat_dec, nan_mask)
            spat_dec = spat_dec.detach().cpu()
            spec_recon_loss = None

        return spat_recon_loss, spec_recon_loss, code_loss, tokens, spat_dec, spec_dec
    
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
        #Encoder
        fused_proj, spat_ids_restore, spec_ids_restore, spat_mask, spec_mask = self.encoder(x, mask)

        # Quantization
        _ , _, tokens = self.quantizer(fused_proj)
        return tokens
    
 
      
if __name__=='__main__':
    # Define configuration as a dictionary
    config = {
        'model': {
            'patch_size': 8,  #Spatial_enc
            'image_size': 32,
            'use_ckpt': False,
            'num_tokens': 1,  #Spectral_enc
            'num_channels': 3,
            'enc_dim': 768
        },
        'quantizer': {
            'quant_type': 'lucid',  # or 'memcodes'
            'latent_dim': 768,
            'codebook_size': 2**16,
            'num_codebooks': 1,
            'norm_codes': True,
            'threshold_ema_dead_code': 2,
            'code_replacement_policy': 'batch_random',
            'sync_codebook': False,
            'ema_decay': 0.99,
            'commitment_weight': 0.25,
            'norm_latents': False,
            'kmeans_init': False
        },
        'debug': False,
        'loss':{'norm_pix_loss':True}
        }

    # Instantiate the model
    model = SpatSpecModel(config)

    # Test the forward pass with a random input
    inp = torch.randn(2, 3, 32,32)
    output = model.tokenize(inp)
    output.shape