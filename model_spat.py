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

def get_crop_size(crop_coords: torch.Tensor) -> torch.Tensor:
    """Returns the crop heights and widths from the crop coordinates."""
    heights = crop_coords[:,2]  # Direct height from the third column
    widths = crop_coords[:,3]   # Direct width from the fourth column
    return torch.stack([heights, widths], dim=1)

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
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
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
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
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
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x

class SpatialModel(nn.Module):
    def __init__(self, config):
        super(SpatialModel, self).__init__()
        self.config = config
        self.perceptloss = None
        self.in_channels = config['model']['num_channels']
        self.image_size = config['model']['image_size']
        self.prediction_type = config['scheduler'].get('prediction_type', 'sample')
        loss_cfg = self.config['loss']
        self.loss_fn_name = loss_cfg.get('fn', 'mse')

        if config['loss']['use_percept']:
            self.perceptloss = TimmPerceptualLoss(
                model_id=config['loss']['model_id'],
                feature_ids=config['loss']['feature_ids'].split('-')
            ).eval().to(device)

        # Load pretrained DiVAE
        pretrained_model_id = 'EPFL-VILAB/4M_tokenizers_rgb_16k_224-448'
        print(f"Loading pretrained DiVAE model from: {pretrained_model_id}")
        self.divae = DiVAE.from_pretrained(pretrained_model_id)
        self.divae.train()
        self.num_train_timesteps = config['scheduler']['num_train_timesteps']

        # Freeze decoder parameters
        for name, param in self.divae.named_parameters():
            if "decoder" in name:
                param.requires_grad = False
                print(f"Froze decoder parameter: {name}")

        # Unfreeze blocks 11 and above in encoder
        for name, param in self.divae.encoder.named_parameters():
            if ("blocks.11" in name or "norm_mlp" in name or "post_mlp" in name):
                param.requires_grad = True
                print(f"Unfroze encoder parameter: {name}")
            else:
                param.requires_grad = False
                print(f"Froze encoder parameter: {name}")

        # Check codebook parameters
        modules_to_check = ['quant_proj', 'quantize', 'cls_emb']
        print("\nChecking parameters in 'quant_proj', 'quantize', and 'cls_emb' for the codebook...\n")
        for module_name in modules_to_check:
            if hasattr(self.divae, module_name):
                module = getattr(self.divae, module_name)
                if module is not None:
                    print(f"\nParameters in module: {module_name}")
                    for param_name, param in module.named_parameters():
                        print(f"{module_name}.{param_name}, requires_grad: {param.requires_grad}, shape: {param.shape}")
                else:
                    print(f"Module {module_name} exists but is None.")
            else:
                print(f"Module {module_name} not found in DiVAE model.")

        # Unfreeze last two decoder layers
        layers_to_unfreeze = ['decoder.out.0', 'decoder.out.2']
        print("\nUnfreezing the last two decoder layers: 'decoder.out.0' and 'decoder.out.2'\n")
        for layer in layers_to_unfreeze:
            for name, param in self.divae.named_parameters():
                if name.startswith(layer):
                    param.requires_grad = True
                    print(f"Unfroze decoder parameter: {name}")

    def forward(self, samples: torch.Tensor, nan_mask: torch.Tensor, crop_coords: torch.Tensor,num_steps=100, **kwargs):
        """
        Forward pass:
        
        samples: (B,1,H,W) typically (B,1,128,128)
        nan_mask: mask tensor
        crop_coords: (B,4) containing (top, left, h, w)

        We'll use crop_coords to compute orig_res=(h,w) and condition the model.
        """
        orig_res = None
        if crop_coords!=None:
            orig_res = get_crop_size(crop_coords).to(device) 

        # Step 1: Upsample and normalize
        x_resized = F.interpolate(samples, size=(224, 224), mode='bilinear', align_corners=False)
        x_upsampled = x_resized.repeat(1, 3, 1, 1)
        mean_rgb = torch.tensor(IMAGENET_INCEPTION_MEAN, device=samples.device).view(1, 3, 1, 1)
        std_rgb = torch.tensor(IMAGENET_INCEPTION_STD, device=samples.device).view(1, 3, 1, 1)
        x_normalized = (x_upsampled - mean_rgb) / std_rgb

        B = x_normalized.size(0)
        # Random timesteps
        timesteps = torch.randint(0, num_steps, (B,), device=x_normalized.device).long()
        noise = torch.randn_like(x_normalized)
        noisy_images = self.divae.noise_scheduler.add_noise(x_normalized, noise, timesteps)

        # Forward through DiVAE with orig_res
        model_output, code_loss = self.divae(x_normalized, noisy_images, timesteps, orig_res=orig_res)

        # Determine target based on prediction_type
        prediction_type = self.divae.prediction_type

        if prediction_type == 'sample':
            target = samples
            model_output = F.interpolate(model_output, size=samples.shape[-2:], mode='area')
            model_output = model_output.mean(dim=1, keepdim=True)
        elif prediction_type == 'epsilon':
            target = noise
        elif prediction_type == 'v_prediction':
            target = noise_scheduler.get_velocity(x_normalized, noise, timesteps)
        elif prediction_type == 'v_prediction-epsilon_loss':
            target = noise
            model_output = noise_scheduler.get_noise(noisy_images, model_output, timesteps)
        else:
            raise ValueError(f"Unknown prediction_type: {prediction_type}")

        final_recon_loss = F.mse_loss(model_output, target)

        # Perceptual loss if needed
        percept_loss_value = 0.0
        if self.perceptloss is not None:
            original_rgb = samples if samples.shape[1]==3 else samples.repeat(1,3,1,1)
            reconstructed_rgb = model_output if model_output.shape[1]==3 else model_output.repeat(1,3,1,1)
            percept_loss_value = self.perceptloss(original_rgb, reconstructed_rgb, preprocess_inputs=True)

        # Combine losses
        commitment_weight = self.config['quantizer'].get('commitment_weight', 1.0)
        total_loss = final_recon_loss + code_loss * commitment_weight
        if self.config['loss'].get('use_percept', False):
            total_loss += percept_loss_value.mean()

        x_downsampled = model_output

        return total_loss, final_recon_loss, percept_loss_value, code_loss, None, x_downsampled, None
    
    def recon_loss(self, input, target, nan_mask):
        # Computes the reconstruction loss (MSE) between input and target, considering nan_mask.
        device = input.device
        target = target.to(device)
        nan_mask = nan_mask.to(device)
        nan_mask = ~nan_mask.bool()
        diff = (input - target) * nan_mask
        loss = torch.sum(diff.pow(2)) / torch.sum(nan_mask)
        return loss

    def tokenize(self, x, mask=0.0):
        # Convert grayscale to RGB if necessary
        if self.in_channels == 1 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Normalize input image
        mean_rgb = torch.tensor(IMAGENET_INCEPTION_MEAN).view(1, 3, 1, 1).to(x.device)
        std_rgb = torch.tensor(IMAGENET_INCEPTION_STD).view(1, 3, 1, 1).to(x.device)
        x_normalized = (x - mean_rgb) / std_rgb

        # Encoder
        quant_enc, code_loss, tokens = self.divae.encode(x_normalized)
        return tokens
# Test write
