# models/model_spat.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import yaml
import os
from PIL import Image
import matplotlib.pyplot as plt

# Relative import for models_mae_Spat
from models.models_mae_Spat import spat_mae_b

# Import fourm classes
from models.ml_4m.fourm.vq.quantizers import VectorQuantizerLucid, Memcodes
from models.ml_4m.fourm.vq.vqvae import DiVAE
from models.percept_loss import TimmPerceptualLoss
from models.ml_4m.fourm.vq.scheduling import DDIMScheduler, PipelineCond

# Import diffusion models from diffusers
from diffusers import UNet2DConditionModel, UNet2DModel

# Import utility functions
from models.ml_4m.fourm.utils import denormalize, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_rank():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0  # Default to 0 if not initialized

def all_reduce_fn(tensor, use_distributed=False):
    # Perform all_reduce only if distributed is initialized and use_distributed is True
    if torch.distributed.is_initialized() and use_distributed:
        torch.distributed.all_reduce(tensor)
    else:
        print("Skipping all_reduce as distributed is not initialized or necessary.")

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with different dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f'p={self.drop_prob}'

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features or in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0., kernel_size=7, padding=3, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma = None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class SpatialModel(nn.Module):
    def __init__(self, config):
        super(SpatialModel, self).__init__()
        self.config = config
        self.perceptloss = None
        self.in_channels = config['model']['num_channels']
        self.image_size = config['model']['image_size']

        if config['loss']['use_percept']:
            self.perceptloss = TimmPerceptualLoss(
                model_id=config['loss']['model_id'],
                feature_ids=config['loss']['feature_ids'].split('-')
            ).eval().to(device)

        # Load the pretrained DiVAE model
        pretrained_model_id = 'EPFL-VILAB/4M_tokenizers_rgb_16k_224-448'
        print(f"Loading pretrained DiVAE model from: {pretrained_model_id}")
        self.divae = DiVAE.from_pretrained(pretrained_model_id)#.to(device)
        self.divae.train()
        self.divae.prediction_type = 'sample'  # Set prediction_type as needed
        self.num_train_timesteps = config['scheduler']['num_train_timesteps']

        # Use the noise scheduler from DiVAE
        self.num_train_timesteps = self.divae.noise_scheduler.config.num_train_timesteps
        print(self.num_train_timesteps)
        # Ensure all parameters are trainable
        # for name, param in self.divae.named_parameters():
        #     param.requires_grad = True
        #         # Freeze decoder parameters
        for name, param in self.divae.named_parameters():
            if "decoder" in name:
                param.requires_grad = False
                print(f"Froze decoder parameter: {name}")

        # Unfreeze blocks 10 and above in the encoder
        for name, param in self.divae.encoder.named_parameters():
            #"blocks.10" in name or
            if (
                "blocks.11" in name or
                "norm_mlp" in name or "post_mlp" in name
            ):
                param.requires_grad = True
                print(f"Unfroze encoder parameter: {name}")
            else:
                param.requires_grad = False
                print(f"Froze encoder parameter: {name}")

        # Check parameters in quant_proj, quantize, and cls_emb
        modules_to_check = ['quant_proj', 'quantize', 'cls_emb']
        print("\nChecking parameters in 'quant_proj', 'quantize', and 'cls_emb' for the codebook...\n")

        for module_name in modules_to_check:
            if hasattr(self.divae, module_name):
                module = getattr(self.divae, module_name)
                if module is not None:  # Check if the module is not None
                    print(f"\nParameters in module: {module_name}")
                    for param_name, param in module.named_parameters():
                        print(f"{module_name}.{param_name}, requires_grad: {param.requires_grad}, shape: {param.shape}")
                else:
                    print(f"Module {module_name} exists but is None.")
            else:
                print(f"Module {module_name} not found in DiVAE model.")
        # --- Unfreeze the Last Two Decoder Layers ---
        # Define the last two decoder layers based on your parameter list
        # From the provided parameter list, the last two layers appear to be 'decoder.out.0' and 'decoder.out.2'
        layers_to_unfreeze = ['decoder.out.0', 'decoder.out.2']

        print("\nUnfreezing the last two decoder layers: 'decoder.out.0' and 'decoder.out.2'\n")

        for layer in layers_to_unfreeze:
            for name, param in self.divae.named_parameters():
                if name.startswith(layer):
                    param.requires_grad = True
                    print(f"Unfroze decoder parameter: {name}")
    def forward(self, x, nan_mask=None, enc_mask=None, use_diffusion=None, timesteps=None, generator=None, verbose=False, **kwargs):
        # x: Input image tensor of shape [batch_size, 1, 128, 128]

        # Step 1: Upsample x to [batch_size, 3, 256, 256] and normalize
        x_resized = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x_upsampled = x_resized.repeat(1, 3, 1, 1)
        mean_rgb = torch.tensor(IMAGENET_INCEPTION_MEAN, device=x.device).view(1, 3, 1, 1)
        std_rgb = torch.tensor(IMAGENET_INCEPTION_STD, device=x.device).view(1, 3, 1, 1)
        x_normalized = (x_upsampled - mean_rgb) / std_rgb

        # Step 2: Generate random timesteps
        if timesteps is None:
            timesteps = torch.randint(0, self.num_train_timesteps, (x.size(0),), device=x.device).long() #, device=device
        else:
            timesteps = timesteps.to(x.device).long()

        # Step 3: Generate noise
        noise = torch.randn_like(x_normalized)

        # Step 4: Add noise to the clean images according to the noise magnitude at each timestep
        noisy_images = self.divae.noise_scheduler.add_noise(x_normalized, noise, timesteps)

        # Step 5: Forward pass through DiVAE model
        # This will internally encode the images and decode
        model_output, code_loss = self.divae(x_normalized, noisy_images, timesteps)

        # Step 6: Determine target based on prediction_type
        if self.divae.prediction_type == 'sample':
            target = x
            #Downsample the reconstructed image to match the original input size
            model_output = F.interpolate(model_output, size=(128, 128), mode='area')
            model_output = model_output.mean(dim=1, keepdim=True)  # Convert to grayscale
            
        elif self.divae.prediction_type == 'epsilon':
            target = noise
        elif self.divae.prediction_type == 'v_prediction':
            target = self.divae.noise_scheduler.get_velocity(x_normalized, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction_type: {self.divae.prediction_type}")

        

        # Step 9: Compute final reconstruction loss between x and x_downsampled
        
        #final_recon_loss = self.recon_loss(target, model_output, nan_mask)
        final_recon_loss = F.mse_loss(target, model_output)

        # Step 10: Compute perceptual loss if applicable
        percept_loss_value = 0.0
        if self.perceptloss is not None:
            # Convert x and x_downsampled to RGB for perceptual loss
            original_rgb = x.repeat(1, 3, 1, 1)
            reconstructed_rgb = x_downsampled.repeat(1, 3, 1, 1)
            percept_loss_value = self.perceptloss(original_rgb, reconstructed_rgb, preprocess_inputs=True)

        # Step 11: Combine losses
        commitment_weight = self.config['quantizer'].get('commitment_weight', 1.0)
        total_loss = final_recon_loss + code_loss * commitment_weight
        if self.config['loss'].get('use_percept', False):
            total_loss += percept_loss_value.mean()

        # Return losses and outputs as needed
        return total_loss, final_recon_loss, percept_loss_value, code_loss, None, model_output, None


    def recon_loss(self, input, target, nan_mask):
        # Computes the reconstruction loss (MSE) between input and target, considering the nan_mask.
        device = input.device
        target = target.to(device)
        nan_mask = nan_mask.to(device)

        # Invert nan_mask to get valid regions
        nan_mask = ~nan_mask.bool()

        # Calculate loss only for non-NaN values
        diff = (input - target) * nan_mask
        loss = torch.sum(diff.pow(2)) / torch.sum(nan_mask)

        return loss

    def tokenize(self, x, mask=0.0):
        # Convert grayscale to RGB if necessary
        if self.in_channels == 1 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Normalize the input image
        mean_rgb = torch.tensor(IMAGENET_INCEPTION_MEAN).view(1, 3, 1, 1).to(x.device)
        std_rgb = torch.tensor(IMAGENET_INCEPTION_STD).view(1, 3, 1, 1).to(x.device)
        x_normalized = (x - mean_rgb) / std_rgb

        # Encoder
        quant_enc, code_loss, tokens = self.divae.encode(x_normalized)
        return tokens
