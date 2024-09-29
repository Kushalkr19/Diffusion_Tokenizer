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
            f'{prefix}total_loss': losses['total_loss'],
            f'{prefix}spat_percept_loss': losses['spat_percept_loss'],
        }
        if losses['spec_recon_loss'] is not None:
            log_dict[f'{prefix}spec_recon_loss'] = losses['spec_recon_loss']
        self.pl_logger.log_metrics(log_dict, step=step)

    def log_learning_rate(self, lr, step):
        self.pl_logger.log_metrics({'learning_rate': lr}, step=step)
        
    def log_metrics(self, metrics, step):
        self.pl_logger.log_metrics(metrics, step=step)

    def sanity_check(self, samples, spat_recon, spec_recon, step, num_plots=10):
        idxs = np.random.randint(0, samples.shape[0], size=num_plots)
        fig, axs = plt.subplots(num_plots, 4, figsize=(40, num_plots * 10))

        for i, idx in enumerate(idxs):
            sample = samples[idx]
            spat_recon_img = spat_recon[idx]

            if len(spat_recon_img.shape) != 3:
                raise ValueError("Reconstructed arrays should have 3 dimensions: (C, H, W)")

            # Original image
            orig_img = np.transpose(sample, (1, 2, 0))
            axs[i, 0].imshow(orig_img[:, :, 0])
            axs[i, 0].set_title('Original', fontsize=20)
            axs[i, 0].axis('off')

            # Spatial reconstruction
            spat_recon_img = np.transpose(spat_recon_img, (1, 2, 0))
            axs[i, 1].imshow(spat_recon_img[:, :, 0])
            axs[i, 1].set_title('Spat Recon', fontsize=20)
            axs[i, 1].axis('off')

            # Difference between original and spatial reconstruction
            diff_img = np.mean(np.abs(orig_img - spat_recon_img), axis=-1)
            axs[i, 2].imshow(diff_img, cmap='inferno')
            axs[i, 2].set_title('Diff (Orig - Spat Recon)', fontsize=20)
            axs[i, 2].axis('off')

            # Spectral reconstruction
            if spec_recon is not None:
                spec_recon_img = spec_recon[idx]
                if len(spec_recon_img.shape) != 3:
                    raise ValueError("Reconstructed arrays should have 3 dimensions: (C, H, W)")
                spec_recon_img = np.transpose(spec_recon_img, (1, 2, 0))
                axs[i, 3].imshow(spec_recon_img[:, :, spec_recon_img.shape[2] // 2], cmap='inferno')
                axs[i, 3].set_title('Spec Recon', fontsize=20)
                axs[i, 3].axis('off')
            else:
                axs[i, 3].axis('off')

        plt.tight_layout()
        self.pl_logger.experiment.add_figure('Sanity Check', fig, global_step=step)
        plt.close(fig)
