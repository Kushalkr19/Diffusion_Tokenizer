import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch
import numpy as np

class MAELogger:
    def __init__(self, pl_logger):
        self.pl_logger = pl_logger

    def log_losses(self, losses, step, prefix=''):
        log_dict = {
            f'{prefix}spat_recon_loss': losses['spat_recon_loss'],
            f'{prefix}code_loss': losses['code_loss'],
            f'{prefix}total_loss': losses['total_loss']
        }
        if losses['spec_recon_loss'] is not None:
            log_dict[f'{prefix}spec_recon_loss'] = losses['spec_recon_loss']
        self.pl_logger.log_metrics(log_dict, step=step)

    def log_learning_rate(self, lr, step):
        self.pl_logger.log_metrics({'learning_rate': lr}, step=step)
        
    def log_metrics(self, metrics, step):
        self.pl_logger.log_metrics(metrics, step=step)

    def sani_check(self, samples, spat_recon, spec_recon, get_spectrogram_func, step):
        idx = torch.randint(0, samples.shape[0], (1,)).item()
        sample = samples[idx].unsqueeze(0)
        spat_recon = spat_recon[idx].unsqueeze(0)

        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        
        images = [
            (sample.squeeze().permute(1, 2, 0), 'Original Spatial Image', False),
            (spat_recon.squeeze().permute(1, 2, 0), 'Reconstructed Spatial Image', False),
            ((sample - spat_recon).squeeze().permute(1, 2, 0), 'Spatial Image Difference', False)
        ]

        if spec_recon is not None:
            spec_recon = spec_recon[idx].unsqueeze(0)
            images.extend([
                (spec_recon, 'Reconstructed Spectrogram', True),
                (orig_spec - spec_recon, 'Spectrogram Difference', True)
            ])

        for i, (img, title, is_spectrogram) in enumerate(images):
            ax = axs[i // 3, i % 3]
            im = ax.imshow(img.cpu().numpy(), 
                           cmap='inferno' if is_spectrogram else None, 
                           origin='lower' if is_spectrogram else None)
            ax.set_title(title)
            ax.axis('off')

        # If spec_recon is None, remove the empty subplots
        if spec_recon is None:
            fig.delaxes(axs[1, 1])
            fig.delaxes(axs[1, 2])

        plt.tight_layout()
        self.pl_logger.experiment.add_figure('Sanity Check', fig, global_step=step)
        plt.close(fig)
        

    def sanity_check(self, samples, spat_recon, spec_recon, step, num_plots=10):
        idxs = np.random.randint(0, samples.shape[0], size=num_plots)
        fig, axs = plt.subplots(num_plots, 4, figsize=(40, num_plots*10))  # Increased figure size

        for i, idx in enumerate(idxs):
            sample = samples[idx]
            spat_recon_img = spat_recon[idx]
            if len(spat_recon_img.shape) != 3:
                raise ValueError("Reconstructed arrays should have 3 dimensions: (C, H, W)")

            # Original image
            orig_img = np.transpose(sample, (1, 2, 0))
            axs[i, 0].imshow(orig_img[:,:,41])
            axs[i, 0].set_title('Original', fontsize=20)  # Increased font size
            axs[i, 0].axis('off')

            # Spatial reconstruction
            spat_recon_img = np.transpose(spat_recon_img, (1, 2, 0))
            axs[i, 1].imshow(spat_recon_img[:,:,41])
            axs[i, 1].set_title('Spat Recon', fontsize=20)  # Increased font size
            axs[i, 1].axis('off')

            # Difference between original and spatial reconstruction
            diff_img = np.mean(np.abs(orig_img - spat_recon_img), axis=-1)
            axs[i, 2].imshow(diff_img)
            axs[i, 2].set_title('Diff (Orig - Spat Recon)', fontsize=20)  # Increased font size
            axs[i, 2].axis('off')

            # Spectral reconstruction
            if spec_recon is not None:
                spec_recon_img = spec_recon[idx]
                if len(spec_recon_img.shape) != 3:
                    raise ValueError("Reconstructed arrays should have 3 dimensions: (C, H, W)")
                spec_recon_img = np.transpose(spec_recon_img, (1, 2, 0))
                axs[i, 3].imshow(spec_recon_img[:,:,41])
                axs[i, 3].set_title('Spec Recon', fontsize=20)  # Increased font size
                axs[i, 3].axis('off')
            else:
                axs[i, 3].axis('off')

        plt.tight_layout()
        self.pl_logger.experiment.add_figure('Sanity Check', fig, global_step=step)  # Increased DPI
        plt.close(fig)