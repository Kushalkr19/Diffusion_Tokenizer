import argparse
import os
from pathlib import Path
import yaml
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

import timm
import timm.optim.optim_factory as optim_factory

from models.hs_util import misc
from models.hs_util.misc import NativeScalerWithGradNormCount as NativeScaler
from models.model_spat_spec import SpatSpecModel
from models.model_spat import SpatialModel
from pytorch_lightning.strategies import DDPStrategy
import time
from logger import MAELogger

from torchmetrics import MeanSquaredError, MeanAbsoluteError, PeakSignalNoiseRatio, MultiScaleStructuralSimilarityIndexMeasure
from dataset import CubeData
from torch.utils.data import random_split
from models.ml_4m.fourm.vq.scheduling.diffusion_pipeline import PipelineCond
from models.ml_4m.fourm.vq.scheduling.scheduling_ddim import DDIMScheduler


class MAEPreTrainer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        if config['model']['num_channels'] in [1, 9]: 
            self.model = SpatialModel(config) 
            print('Training Spatial Model Only!!!')

            # Move metrics to the device
            self.mse_metric = MeanSquaredError().to(self.device)
            self.mae_metric = MeanAbsoluteError().to(self.device)
            self.psnr_metric = PeakSignalNoiseRatio().to(self.device)
        else:
            self.model = SpatSpecModel(config)  # CIFAR-10 has 3 channels

        self.loss_scaler = NativeScaler()
        self.mae_logger = None

    def on_fit_start(self):
        # Initialize the logger when the fit starts
        self.mae_logger = MAELogger(self.logger)

    def training_step(self, batch, batch_idx):
        samples, nan_mask = batch
        samples = samples.to(self.device)
        nan_mask = nan_mask.to(self.device)
        # Adjusted to match the outputs from SpatialModel's forward method
        total_loss, final_recon_loss, percept_loss_value, code_loss, _, x_downsampled, _ = self.model(samples, nan_mask)
        
        # Assign outputs
        spat_recon_loss = final_recon_loss
        spat_percept_loss = percept_loss_value
        spec_recon_loss = None  # Since we are only training the spatial model
        spat_recon = x_downsampled
        spec_recon = None  # Not used

        losses = {
            'spat_recon_loss': spat_recon_loss.item(),
            'spec_recon_loss': spec_recon_loss,
            'spat_percept_loss': spat_percept_loss.item() if isinstance(spat_percept_loss, torch.Tensor) else 0.0,
            'code_loss': code_loss.item(),
            'total_loss': total_loss.item()
        }

        if self.trainer.is_global_zero:
            self.mae_logger.log_losses(losses, self.global_step)
            lr = self.optimizers().param_groups[0]['lr']
            self.mae_logger.log_learning_rate(lr, self.global_step)

            if self.global_step % self.config['training']['sanity_check_frequency'] == 0:
                # Handle None cases
                spec_recon_numpy = spec_recon.detach().cpu().numpy() if spec_recon is not None else None
                self.mae_logger.sanity_check(
                    samples.detach().cpu().numpy(), 
                    spat_recon.detach().cpu().numpy(),
                    spec_recon_numpy, 
                    self.global_step
                )

        return total_loss

    def validation_step(self, batch, batch_idx):
        samples, nan_mask = batch
        samples = samples.to(self.device)
        nan_mask = nan_mask.to(self.device)
        # Adjusted to match the outputs from SpatialModel's forward method
        total_loss, final_recon_loss, percept_loss_value, code_loss, _, x_downsampled, _ = self.model(samples, nan_mask)
        
        # Assign outputs
        spat_recon_loss = final_recon_loss
        spat_percept_loss = percept_loss_value
        spec_recon_loss = None  # Not used
        recon = x_downsampled

        losses = {
            'spat_recon_loss': spat_recon_loss.item(),
            'spec_recon_loss': spec_recon_loss,
            'spat_percept_loss': spat_percept_loss.item() if isinstance(spat_percept_loss, torch.Tensor) else 0.0,
            'code_loss': code_loss.item(),
            'total_loss': total_loss.item()
        }

        if self.trainer.is_global_zero:
            self.mae_logger.log_losses(losses, self.global_step, prefix='val_')
        self.log('val_loss', total_loss) 
        
        self.mse_metric(recon, samples.detach())
        self.mae_metric(recon, samples.detach())
        self.psnr_metric(recon, samples.detach())
        
        return total_loss
    
    def on_validation_epoch_end(self):
        # Compute and log metrics
        metrics = {
            'val_MSE': self.mse_metric.compute().item(),
            'val_MAE': self.mae_metric.compute().item(),
            'val_PSNR': self.psnr_metric.compute().item(),
        }

        # Log metrics
        if self.trainer.is_global_zero:
            self.mae_logger.log_metrics(metrics, self.global_step)

        # Reset metrics
        self.mse_metric.reset()
        self.mae_metric.reset()
        self.psnr_metric.reset()

    def configure_optimizers(self):
        param_groups = optim_factory.param_groups_weight_decay(
            self.model, self.config['training']['weight_decay']
        )
        optimizer = torch.optim.AdamW(
            param_groups, 
            lr=self.config['training']['lr'], 
            betas=(0.9, 0.95)
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.config['training']['epochs'], 
            eta_min=self.config['training']['min_lr']
        )
        
        return [optimizer], [scheduler]
    

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main(config_path):
    config = load_config(config_path)
    print('Loaded Config')
    
    pl.seed_everything(config['system']['seed'])

    dataset = CubeData(
        root=config['data']['root'],
        name=config['data']['name'],
        folder=config['data']['folder'],
        file=config['data']['file']
    )

    # Split dataset into train and validation sets
    train_size = int(0.80 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print('The len of train_dataset: ', len(train_dataset))
    print('The len of val_dataset: ', len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_mem'],
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['testing']['batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_mem'],
        shuffle=False,
    )

    model = MAEPreTrainer(config)

    checkpoint_callback = ModelCheckpoint(
        dirpath=config['output']['dir'],
        filename='mae-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
        monitor='val_loss'
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    logger = TensorBoardLogger(save_dir=config['output']['log_dir'], name="mae_logs")
    print('The GPU is available ', torch.cuda.is_available(), 'We have', torch.cuda.device_count())

    trainer = pl.Trainer(
        max_epochs=config['training']['epochs'],
        accelerator="gpu", 
        devices=config['system']['gpu_num'], 
        num_nodes=config['system']['num_nodes'],
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=10,
        accumulate_grad_batches=config['training']['accum_iter'],
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAE pre-training')

    parser.add_argument('--config', type=str, default='./config_cifar.yaml', help='path to config file')
    args = parser.parse_args()

    config = load_config(args.config)
    if config['output']['dir']:
        Path(config['output']['dir']).mkdir(parents=True, exist_ok=True)
    start_time = time.time() 
    main(args.config)
    print('The time taken is ========>', time.time()-start_time)
