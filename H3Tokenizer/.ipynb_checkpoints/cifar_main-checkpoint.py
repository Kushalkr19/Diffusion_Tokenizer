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

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from model_spat_spec import SpatSpecModel
from pytorch_lightning.strategies import DDPStrategy
import time

class MAEPreTrainer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model =  SpatSpecModel(config) # CIFAR-10 has 3 channels
        self.loss_scaler = NativeScaler()

    def training_step(self, batch, batch_idx):
        samples, _ = batch
        loss, pred, mask = self.model(samples, mask_ratio=self.config['model']['mask_ratio'])
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Log learning rate
        lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        # Run sanity check every N steps
        if self.global_step % self.config['training']['sanity_check_frequency'] == 0:
            sanity_check(samples, pred, mask)

        return loss

    def validation_step(self, batch, batch_idx):
        samples, _ = batch
        loss, _, _ = self.model(samples, mask_ratio=self.config['model']['mask_ratio'])
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        param_groups = optim_factory.param_groups_weight_decay(self.model, self.config['training']['weight_decay'])
        optimizer = torch.optim.AdamW(param_groups, lr=self.config['training']['lr'], betas=(0.9, 0.95))
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.config['training']['epochs'], 
            eta_min=self.config['training']['min_lr']
        )
        
        return [optimizer], [scheduler]


    def sanity_check(self, samples, pred, mask):
        # Select a random sample from the batch
        idx = torch.randint(0, samples.shape[0], (1,)).item()
        sample = samples[idx].unsqueeze(0)
    

        # Get the reconstructed image
        im_masked = self.model.unpatchify(pred)
        im_masked = im_masked[idx].unsqueeze(0)

        # Visualize the results
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
         
        # Original image
        orig_img = sample.squeeze().permute(1, 2, 0).cpu().numpy()
        axs[0].imshow(orig_img)
        axs[0].set_title('Original')
        axs[0].axis('off')

        # Masked image
        # Reshape mask to match image dimensions
        mask = mask.unsqueeze(-1).repeat(1, 1, torch.prod(torch.tensor(samples.shape[-3:])).item()//self.config['model']['patch_size']**2)
        mask_reshaped = self.model.unpatchify(mask)[idx].permute(1,2,0).cpu().numpy()
        
        
        masked_img = orig_img * (1 - mask_reshaped)
        axs[1].imshow(masked_img)
        axs[1].set_title('Masked')
        axs[1].axis('off')

        # Reconstructed image
        recon_img = im_masked.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        axs[2].imshow(recon_img)
        axs[2].set_title('Reconstructed')
        axs[2].axis('off')

        plt.tight_layout()
        
        # Log the figure to TensorBoard
        self.logger.experiment.add_figure('Sanity Check', fig, global_step=self.global_step)
        plt.close(fig)
        
        
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main(config_path):
    config = load_config(config_path)
    
    pl.seed_everything(config['system']['seed'])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
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