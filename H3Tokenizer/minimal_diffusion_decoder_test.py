import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, DDIMScheduler
from einops import rearrange

# Define a simplified SpatialModel
class MinimalSpatialModel(nn.Module):
    def __init__(self, device='cuda'):
        super(MinimalSpatialModel, self).__init__()
        self.device = device
        
        # Define a lightweight UNet2DConditionModel
        self.diffusion_model = UNet2DConditionModel(
            sample_size=32,           # Smaller image size for testing
            in_channels=32,           # Adjusted number of input channels to 32
            out_channels=32,          # Adjusted number of output channels to 32
            layers_per_block=1,       # Fewer layers per block
            block_out_channels=(32, 32),  # Adjusted to be divisible by 32
            down_block_types=(
                "DownBlock2D",  # 32x32 -> 16x16
                "DownBlock2D",  # 16x16 -> 8x8
            ),
            up_block_types=(
                "UpBlock2D",    # 8x8 -> 16x16
                "UpBlock2D",    # 16x16 -> 32x32
            ),
            class_embed_type=None,
            cross_attention_dim=32,
        ).to(device)
        
        # Define a lightweight DDIMScheduler
        self.scheduler = DDIMScheduler(
            num_train_timesteps=50,  # Fewer timesteps for testing
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
        
    def diffusion_decoder(self, quant_enc, timesteps=10, generator=None, image_size=32, verbose=False):
        """
        Simplified diffusion_decoder for testing.
        """
        if timesteps is None:
            timesteps = 10

        # Set scheduler timesteps
        self.scheduler.set_timesteps(timesteps)

        if generator is None:
            generator = torch.manual_seed(42)  # For reproducibility

        # Ensure quant_enc is on the correct device
        quant_enc = quant_enc.to(self.device)

        # Reshape quant_enc from [B, C, H, W] to [B, S, C]
        B, C, H, W = quant_enc.shape
        cond = rearrange(quant_enc, 'b c h w -> b (h w) c')  # [B, S, C]

        if verbose:
            print("Before pipeline:")
            print(f"cond shape: {cond.shape}")  # Should be [B, S, C]

        # Simulate the diffusion pipeline by passing cond through the diffusion model
        with torch.no_grad():
            # Randomly select timesteps for each sample in the batch
            timestep = torch.randint(0, timesteps, (B,), device=self.device)
            decoded_image = self.diffusion_model(
                sample=quant_enc,
                timestep=timestep,
                encoder_hidden_states=cond,
            ).sample  # [B, C, H, W]

        if verbose:
            print("After pipeline:")
            print(f"decoded_image shape: {decoded_image.shape}")

        return decoded_image

# Function to perform the test
def test_diffusion_decoder():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize the minimal model
    model = MinimalSpatialModel(device=device)

    # Create a small input tensor
    batch_size = 1
    channels = 32  # Adjusted to 32 to match in_channels
    height = 32
    width = 32
    quant_enc = torch.randn(batch_size, channels, height, width).to(device)

    # Run the diffusion_decoder
    decoded_image = model.diffusion_decoder(quant_enc, timesteps=10, verbose=True)

    print("Decoded Image Tensor:", decoded_image)
    print("Decoded Image Shape:", decoded_image.shape)

if __name__ == "__main__":
    test_diffusion_decoder()
