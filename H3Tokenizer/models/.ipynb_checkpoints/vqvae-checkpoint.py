# Copyright 2024 EPFL and Apple Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Tuple, Dict, Optional, Union, Any
from contextlib import nullcontext
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
#from diffusers.schedulers.scheduling_utils import SchedulerMixin
#from diffusers import StableDiffusionPipeline
from huggingface_hub import PyTorchModelHubMixin

from .quantizers import VectorQuantizerLucid, Memcodes
#import fourm.vq.models.vit_models as vit_models 
#from .models_mae_Spat import spat_mae_b as vit_models
#import fourm.vq.models.unet.unet as unet
#import fourm.vq.models.uvit as uvit
#import fourm.vq.models.controlnet as controlnet
#from fourm.vq.models.mlp_models import build_mlp
#from fourm.vq.scheduling import DDPMScheduler, DDIMScheduler, PNDMScheduler, PipelineCond

#from fourm.utils import denormalize


# If freeze_enc is True, the following modules will be frozen
FREEZE_MODULES = ['encoder', 'quant_proj', 'quantize', 'cls_emb']

class VQ(nn.Module, PyTorchModelHubMixin):
    """Base class for VQVAE and DiVAE models. Implements the encoder and quantizer, and can be used as such without a decoder
    after training.

    Args:
        image_size: Input and target image size.
        image_size_enc: Input image size for the encoder. Defaults to image_size. Change this when loading weights 
          from a tokenizer trained on a different image size.
        n_channels: Number of input channels.
        n_labels: Number of classes for semantic segmentation.
        enc_type: String identifier specifying the encoder architecture. See vq/vit_models.py and vq/mlp_models.py 
            for available architectures.
        patch_proj: Whether or not to use a ViT-style patch-wise linear projection in the encoder.
        post_mlp: Whether or not to add a small point-wise MLP before the quantizer.
        patch_size: Patch size for the encoder.
        quant_type: String identifier specifying the quantizer implementation. Can be 'lucid', or 'memcodes'.
        codebook_size: Number of codebook entries.
        num_codebooks: Number of "parallel" codebooks to use. Only relevant for 'lucid' and 'memcodes' quantizers.
          When using this, the tokens will be of shape B N_C H_Q W_Q, where N_C is the number of codebooks.
        latent_dim: Dimensionality of the latent code. Can be small when using norm_codes=True, 
          see ViT-VQGAN (https://arxiv.org/abs/2110.04627) paper for details.
        norm_codes: Whether or not to normalize the codebook entries to the unit sphere.
          See ViT-VQGAN (https://arxiv.org/abs/2110.04627) paper for details.
        norm_latents: Whether or not to normalize the latent codes to the unit sphere for computing commitment loss.
        sync_codebook: Enable this when training on multiple GPUs, and disable for single GPUs, e.g. at inference.
        ema_decay: Decay rate for the exponential moving average of the codebook entries.
        threshold_ema_dead_code: Threshold for replacing stale codes that are used less than the 
          indicated exponential moving average of the codebook entries.
        code_replacement_policy: Policy for replacing stale codes. Can be 'batch_random' or 'linde_buzo_gray'.
        commitment_weight: Weight for the quantizer commitment loss.
        kmeans_init: Whether or not to initialize the codebook entries with k-means clustering.
        ckpt_path: Path to a checkpoint to load the model weights from.
        ignore_keys: List of keys to ignore when loading the state_dict from the above checkpoint.
        freeze_enc: Whether or not to freeze the encoder weights. See FREEZE_MODULES for the list of modules.
        undo_std: Whether or not to undo any ImageNet standardization and transform the images to [-1,1] 
          before feeding the input to the encoder.
        config: Dictionary containing the model configuration. Only used when loading
            from Huggingface Hub. Ignore otherwise.
    """
    def __init__(self,
                 image_size: int = 224,
                 image_size_enc: Optional[int] = None,
                 n_channels: str = 3,
                 n_labels: Optional[int] = None,
                 enc_type: str = 'vit_b_enc',
                 patch_proj: bool = True,
                 post_mlp: bool = False,
                 patch_size: int = 16,
                 quant_type: str = 'lucid',
                 codebook_size: Union[int, str] = 16384,
                 num_codebooks: int = 1,
                 latent_dim: int = 32,
                 norm_codes: bool = True,
                 norm_latents: bool = False,
                 sync_codebook: bool = True,
                 ema_decay: float = 0.99,
                 threshold_ema_dead_code: float = 0.25,
                 code_replacement_policy: str = 'batch_random',
                 commitment_weight: float = 1.0,
                 kmeans_init: bool = False,
                 ckpt_path: Optional[str] = None,
                 ignore_keys: List[str] = [
                    'decoder', 'loss', 
                    'post_quant_conv', 'post_quant_proj', 
                    'encoder.pos_emb',
                 ],
                 freeze_enc: bool = False,
                 undo_std: bool = False,
                 config: Optional[Dict[str, Any]] = None,
                 **kwargs):
        if config is not None:
            config = copy.deepcopy(config)
            self.__init__(**config)
            return
        
        super().__init__()

        self.image_size = image_size
        self.n_channels = n_channels
        self.n_labels = n_labels
        self.enc_type = enc_type
        self.patch_proj = patch_proj
        self.post_mlp = post_mlp
        self.patch_size = patch_size
        self.quant_type = quant_type
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.latent_dim = latent_dim
        self.norm_codes = norm_codes
        self.norm_latents = norm_latents
        self.sync_codebook = sync_codebook
        self.ema_decay = ema_decay
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.code_replacement_policy = code_replacement_policy
        self.commitment_weight = commitment_weight
        self.kmeans_init = kmeans_init
        self.ckpt_path = ckpt_path
        self.ignore_keys = ignore_keys
        self.freeze_enc = freeze_enc
        self.undo_std = undo_std

        # For semantic segmentation
        if n_labels is not None:
            self.cls_emb = nn.Embedding(num_embeddings=n_labels, embedding_dim=n_channels)
            self.colorize = torch.randn(3, n_labels, 1, 1)
        else:
            self.cls_emb = None

        # Init encoder
        image_size_enc = image_size_enc or image_size
        if 'vit' in enc_type:
            self.encoder = getattr(vit_models, enc_type)(
                in_channels=n_channels, patch_size=patch_size, 
                resolution=image_size_enc, patch_proj=patch_proj, post_mlp=post_mlp
            )
            self.enc_dim = self.encoder.dim_tokens
        elif 'MLP' in enc_type:
            self.encoder = build_mlp(model_id=enc_type, dim_in=n_channels, dim_out=None)
            self.enc_dim = self.encoder.dim_out
        else:
            raise NotImplementedError(f'{enc_type} not implemented.')
        
        # Encoder -> quantizer projection
        self.quant_proj = torch.nn.Conv2d(self.enc_dim, self.latent_dim, 1)

        # Init quantizer
        if quant_type == 'lucid':
            self.quantize = VectorQuantizerLucid(
                dim=latent_dim,
                codebook_size=codebook_size,
                codebook_dim=latent_dim,
                heads=num_codebooks,
                use_cosine_sim = norm_codes,
                threshold_ema_dead_code = threshold_ema_dead_code,
                code_replacement_policy=code_replacement_policy,
                sync_codebook = sync_codebook,
                decay = ema_decay,
                commitment_weight=self.commitment_weight,
                norm_latents = norm_latents,
                kmeans_init=kmeans_init,
            )
        elif quant_type == 'memcodes':
            self.quantize = Memcodes(
                dim=latent_dim, codebook_size=codebook_size,
                heads=num_codebooks, temperature=1.,
            )
        else:
            raise ValueError(f'{quant_type} not a valid quant_type.')

        # Load checkpoint
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        
        # Freeze encoder
        if freeze_enc:
            for module_name, module in self.named_children():
                if module_name not in FREEZE_MODULES:
                    continue
                for param in module.parameters():
                    param.requires_grad = False
                module.eval()

    def train(self, mode: bool = True) -> 'VQ':
        """Override the default train() to set the training mode to all modules 
        except the encoder if freeze_enc is True.

        Args:
            mode: Whether to set the model to training mode (True) or evaluation mode (False).
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module_name, module in self.named_children():
            if self.freeze_enc and module_name in FREEZE_MODULES:
                continue
            module.train(mode)
        return self

    def init_from_ckpt(self, path: str, ignore_keys: List[str] = list()) -> 'VQ':
        """Loads the state_dict from a checkpoint file and initializes the model with it.
        Renames the keys in the state_dict if necessary (e.g. when loading VQ-GAN weights).

        Args:
            path: Path to the checkpoint file.
            ignore_keys: List of keys to ignore when loading the state_dict.

        Returns:
            self
        """
        ckpt = torch.load(path, map_location="cpu")
        sd = ckpt['model'] if 'model' in ckpt else ckpt['state_dict']

        # Compatibility with ViT-VQGAN weights
        if 'quant_conv.0.weight' in sd and 'quant_conv.0.bias' in sd:
            print("Renaming quant_conv.0 to quant_proj")
            sd['quant_proj.weight'] = sd['quant_conv.0.weight']
            sd['quant_proj.bias'] = sd['quant_conv.0.bias']
            del sd['quant_conv.0.weight']
            del sd['quant_conv.0.bias']
        elif 'quant_conv.weight' in sd and 'quant_conv.bias' in sd:
            print("Renaming quant_conv to quant_proj")
            sd['quant_proj.weight'] = sd['quant_conv.weight']
            sd['quant_proj.bias'] = sd['quant_conv.bias']
            del sd['quant_conv.weight']
            del sd['quant_conv.bias']
        if 'post_quant_conv.0.weight' in sd and 'post_quant_conv.0.bias' in sd:
            print("Renaming post_quant_conv.0 to post_quant_proj")
            sd['post_quant_proj.weight'] = sd['post_quant_conv.0.weight']
            sd['post_quant_proj.bias'] = sd['post_quant_conv.0.bias']
            del sd['post_quant_conv.0.weight']
            del sd['post_quant_conv.0.bias']
        elif 'post_quant_conv.weight' in sd and 'post_quant_conv.bias' in sd:
            print("Renaming post_quant_conv to post_quant_proj")
            sd['post_quant_proj.weight'] = sd['post_quant_conv.weight']
            sd['post_quant_proj.bias'] = sd['post_quant_conv.bias']
            del sd['post_quant_conv.weight']
            del sd['post_quant_conv.bias']

        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        msg = self.load_state_dict(sd, strict=False)
        print(msg)
        print(f"Restored from {path}")

        return self

    def prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocesses the input image tensor before feeding it to the encoder.
        If self.undo_std, the input is first denormalized from the ImageNet 
        standardization to [-1, 1]. If semantic segmentation is performed, the 
        class indices are embedded.

        Args:
            x: Input image tensor of shape B C H W 
              or B H W in case of semantic segmentation

        Returns:
            Preprocessed input tensor of shape B C H W
        """
        if self.undo_std:
            x = 2.0 * denormalize(x) - 1.0
        if self.cls_emb is not None:
            x = rearrange(self.cls_emb(x), 'b h w c -> b c h w')
        return x

    def to_rgb(self, x: torch.Tensor) -> torch.Tensor:
        """When semantic segmentation is performed, this function converts the 
        class embeddings to RGB.

        Args:
            x: Input tensor of shape B C H W

        Returns:
            RGB tensor of shape B C H W
        """
        x = F.conv2d(x, weight=self.colorize)
        x = (x-x.min())/(x.max()-x.min())
        return x

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        """Encodes an input image tensor and quantizes the latent code.

        Args:
            x: Input image tensor of shape B C H W
              or B H W in case of semantic segmentation
        
        Returns:
            quant: Quantized latent code of shape B D_Q H_Q W_Q
            code_loss: Codebook loss
            tokens: Quantized indices of shape B H_Q W_Q
        """
        x = self.prepare_input(x)
        h = self.encoder(x)
        h = self.quant_proj(h)
        quant, code_loss, tokens = self.quantize(h)
        return quant, code_loss, tokens

    def tokenize(self, x: torch.Tensor) -> torch.LongTensor:
        """Tokenizes an input image tensor.

        Args:
            x: Input image tensor of shape B C H W
              or B H W in case of semantic segmentation

        Returns:
            Quantized indices of shape B H_Q W_Q
        """
        _, _, tokens = self.encode(x)
        return tokens
    
    def autoencode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Autoencodes an input image tensor by encoding it, quantizing the latent code, 
        and decoding it back to an image.

        Args:
            x: Input image tensor of shape B C H W
              or B H W in case of semantic segmentation
        
        Returns:
            Reconstructed image tensor of shape B C H W
        """
        pass

    def decode_quant(self, quant: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decodes quantized latent codes back to an image.

        Args:
            quant: Quantized latent code of shape B D_Q H_Q W_Q

        Returns:
            Decoded image tensor of shape B C H W
        """
        pass

    def tokens_to_embedding(self, tokens: torch.LongTensor) -> torch.Tensor:
        """Look up the codebook entries corresponding the discrete tokens.

        Args:
            tokens: Quantized indices of shape B H_Q W_Q

        Returns:
            Quantized latent code of shape B D_Q H_Q W_Q
        """
        return self.quantize.indices_to_embedding(tokens)

    def decode_tokens(self, tokens: torch.LongTensor, **kwargs) -> torch.Tensor:
        """Decodes discrete tokens back to an image.

        Args:
            tokens: Quantized indices of shape B H_Q W_Q

        Returns:
            Decoded image tensor of shape B C H W
        """
        quant = self.tokens_to_embedding(tokens)
        dec = self.decode_quant(quant, **kwargs)
        return dec
    
    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the encoder and quantizer.

        Args:
            x: Input image tensor of shape B C H W
              or B H W in case of semantic segmentation
        
        Returns:
            quant: Quantized latent code of shape B D_Q H_Q W_Q
            code_loss: Codebook loss
        """
        quant, code_loss, _ = self.encode(x)
        return quant, code_loss


class VQVAE(VQ):
    """VQ-VAE model = simple encoder + decoder with a discrete bottleneck and 
    basic reconstruction loss (optionall with perceptual loss), i.e. no diffusion, 
    nor GAN discriminator.

    Args:
        dec_type: String identifier specifying the decoder architecture. 
          See vq/vit_models.py and vq/mlp_models.py for available architectures.
        out_conv: Whether or not to add final conv layers to the ViT decoder.
        image_size_dec: Image size for the decoder. Defaults to self.image_size. 
          Change this when loading weights from a tokenizer decoder trained on a 
          different image size.
        patch_size_dec: Patch size for the decoder. Defaults to self.patch_size.
        config: Dictionary containing the model configuration. Only used when loading
            from Huggingface Hub. Ignore otherwise.
    """
    def __init__(self, 
                 dec_type: str = 'vit_b_dec', 
                 out_conv: bool = False,
                 image_size_dec: int = None, 
                 patch_size_dec: int = None,
                 config: Optional[Dict[str, Any]] = None,
                 *args, 
                 **kwargs):
        if config is not None:
            config = copy.deepcopy(config)
            self.__init__(**config)
            return
        # Don't want to load the weights just yet
        self.original_ckpt_path = kwargs.get('ckpt_path', None)
        kwargs['ckpt_path'] = None
        super().__init__(*args, **kwargs)
        self.ckpt_path = self.original_ckpt_path

        # Init decoder
        out_channels = self.n_channels if self.n_labels is None else self.n_labels
        image_size_dec = image_size_dec or self.image_size
        patch_size = patch_size_dec or self.patch_size
        if 'vit' in dec_type:
            self.decoder = getattr(vit_models, dec_type)(
                out_channels=out_channels, patch_size=patch_size, 
                resolution=image_size_dec, out_conv=out_conv, post_mlp=self.post_mlp,
                patch_proj=self.patch_proj
            )
            self.dec_dim = self.decoder.dim_tokens
        elif 'MLP' in dec_type:
            self.decoder = build_mlp(model_id=dec_type, dim_in=None, dim_out=out_channels)
            self.dec_dim = self.decoder.dim_in
        else:
            raise NotImplementedError(f'{dec_type} not implemented.')

        # Quantizer -> decoder projection
        self.post_quant_proj = torch.nn.Conv2d(self.latent_dim, self.dec_dim, 1)

        # Load checkpoint
        if self.ckpt_path is not None:
            self.init_from_ckpt(self.ckpt_path, ignore_keys=self.ignore_keys)

    def decode_quant(self, quant: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decodes quantized latent codes back to an image.

        Args:
            quant: Quantized latent code of shape B D_Q H_Q W_Q

        Returns:
            Decoded image tensor of shape B C H W
        """
        quant = self.post_quant_proj(quant)
        dec = self.decoder(quant)
        return dec

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the encoder, quantizer, and decoder.

        Args:
            x: Input image tensor of shape B C H W
              or B H W in case of semantic segmentation
        
        Returns:
            dec: Decoded image tensor of shape B C H W
            code_loss: Codebook loss
        """
        with torch.no_grad() if self.freeze_enc else nullcontext():
            quant, code_loss, _ = self.encode(x)
        dec = self.decode_quant(quant)
        return dec, code_loss

    def autoencode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Autoencodes an input image tensor by encoding it, quantizing the 
        latent code, and decoding it back to an image.

        Args:
            x: Input image tensor of shape B C H W
              or B H W in case of semantic segmentation
        
        Returns:
            Reconstructed image tensor of shape B C H W
        """
        dec, _ = self.forward(x)
        return dec


