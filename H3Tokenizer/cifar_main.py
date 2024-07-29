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
from pytorch_lightning.strategies import DDPStrategy
import time
from logger import MAELogger

from torchmetrics import MeanSquaredError, MeanAbsoluteError, PeakSignalNoiseRatio, MultiScaleStructuralSimilarityIndexMeasure
#from torchmetrics.image.fid import FrechetInceptionDistance
#from torchmetrics.image.lpips import LearnedPerceptualImagePatchSimilarity
#from torchmetrics.image.inception import InceptionScore

class MAEPreTrainer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model =  SpatSpecModel(config) # CIFAR-10 has 3 channels
        self.loss_scaler = NativeScaler()
        self.mae_logger = None
        self.mse_metric = MeanSquaredError(squared=True)
        self.mae_metric = MeanAbsoluteError()
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0)

    def on_fit_start(self):
        # Initialize the logger when the fit starts
        self.mae_logger = MAELogger(self.logger)

    def training_step(self, batch, batch_idx):
        samples, _ = batch
        spat_recon_loss, spec_recon_loss, code_loss, _, spat_recon, spec_recon = self.model(samples)
        spat_recon = spat_recon.numpy()
        
        if spec_recon is None:
            total_loss = spat_recon_loss + code_loss
        else:
            spec_recon = spec_recon.numpy()
            total_loss = 0.5 * spat_recon_loss + 0.5 * spec_recon_loss + code_loss

        losses = {
            'spat_recon_loss': spat_recon_loss,
            'spec_recon_loss': spec_recon_loss,
            'code_loss': code_loss,
            'total_loss': total_loss
        }

        if self.trainer.is_global_zero:
            self.mae_logger.log_losses(losses, self.global_step)
            lr = self.optimizers().param_groups[0]['lr']
            self.mae_logger.log_learning_rate(lr, self.global_step)

            if self.global_step % self.config['training']['sanity_check_frequency'] == 0:
                self.mae_logger.sanity_check(samples.detach().cpu().numpy(), spat_recon, spec_recon, self.global_step) 

        return total_loss

    def validation_step(self, batch, batch_idx):
        samples, _ = batch
        spat_recon_loss, spec_recon_loss, code_loss,  _,recon,_ = self.model(samples)
        
        if spec_recon_loss is None:
            total_loss = spat_recon_loss + code_loss
        else:
            total_loss = 0.5 * spat_recon_loss + 0.5 * spec_recon_loss + code_loss
            
        losses = {
            'spat_recon_loss': spat_recon_loss,
            'spec_recon_loss': spec_recon_loss,
            'code_loss': code_loss,
            'total_loss': total_loss
        }
        if self.trainer.is_global_zero:
            self.mae_logger.log_losses(losses, self.global_step, prefix='val_')
        self.log('val_loss', total_loss) 
        
        
        self.mse_metric(recon, samples.detach().cpu())
        self.mae_metric(recon, samples.detach().cpu())
        self.psnr_metric(recon, samples.detach().cpu())
        #self.ms_ssim_metric(recon, samples.detach().cpu())
        #self.fid_metric(samples.detach().cpu(), real=True)
        #self.fid_metric(recon.clamp(0, 1), real=False)
        
        return total_loss
    
    def on_validation_epoch_end(self):
        # Compute and log metrics
        metrics = {
            'val_MSE': self.mse_metric.compute(),
            'val_MAE': self.mae_metric.compute(),
            'val_PSNR': self.psnr_metric.compute(),
            #'val_MS-SSIM': self.ms_ssim_metric.compute(),
            #'val_FID': self.fid_metric.compute(),
        }
        
        #inception_mean, inception_std = self.inception_metric.compute()
        #metrics['val_InceptionMean'] = inception_mean
        #metrics['val_InceptionStd'] = inception_std

        # Log metrics
        if self.trainer.is_global_zero:
            self.mae_logger.log_metrics(metrics, self.global_step)

        # Reset metrics
        self.mse_metric.reset()
        self.mae_metric.reset()
        self.psnr_metric.reset()
        #self.ms_ssim_metric.reset()
        #self.fid_metric.reset()
        #self.lpips_metric.reset()
        #self.inception_metric.reset()

    def configure_optimizers(self):
        param_groups = optim_factory.param_groups_weight_decay(self.model, self.config['training']['weight_decay'])
        optimizer = torch.optim.AdamW(param_groups, lr=self.config['training']['lr'], betas=(0.9, 0.95))
        
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
    
    pl.seed_everything(config['system']['seed'])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    dataset_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    dataset_val = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)

    train_loader = DataLoader(
        dataset_train,
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_mem'],
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset_val,
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_mem'],
        shuffle=False,
    )

    model = MAEPreTrainer(config)

    checkpoint_callback = ModelCheckpoint(
        dirpath=config['output']['dir'],
        filename='mae-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
        monitor='val_loss'
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    

    logger = TensorBoardLogger(save_dir=config['output']['log_dir'], name="mae_logs")

    trainer = pl.Trainer(
        max_epochs=config['training']['epochs'],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=config['system']['gpu_num'] if torch.cuda.is_available() else None,
        #strategy = 'ddp' if config['system']['gpu_num'] > 1 else 'auto',
        strategy = DDPStrategy(find_unused_parameters=True),
        #precision=64,  # Using mixed precision
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
    print('The time taken is ========>',time.time()-start_time)