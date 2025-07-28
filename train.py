import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
import os
import json
from tqdm import tqdm
import wandb
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score

from model.models import DeepEVDModel
from utils import EpidemicDataset, calculate_epidemiological_metrics, setup_logging, save_checkpoint, load_checkpoint

class DeepEVDTrainer:
    """
    Trainer class for the DeepEVD model with comprehensive training pipeline
    """
    def __init__(self, config: Dict, use_wandb: bool = False):
        self.config = config
        self.use_wandb = use_wandb
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
        setup_logging(config.get('log_level', 'INFO'))
        self.logger = logging.getLogger(__name__)

        self.model = DeepEVDModel(config['model']).to(self.device)
        self.logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")

        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

        self.loss_weights = config.get('loss_weights', {'case': 1.0, 'r0': 0.5, 'risk': 0.3})
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = {'train': [], 'val': []}
 
        self.patience = config.get('patience', 20)
        self.early_stop_counter = 0
 
        if self.use_wandb:
            wandb.init(project="deepevd", config=config)
            wandb.watch(self.model)
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer with different learning rates for different components"""
        optimizer_config = self.config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adamw')
        base_lr = optimizer_config.get('lr', 1e-3)
        weight_decay = optimizer_config.get('weight_decay', 1e-4)
        param_groups = [
            {'params': self.model.embedding_layer.parameters(), 'lr': base_lr * 0.5},
            {'params': self.model.gcn_layers.parameters(), 'lr': base_lr},
            {'params': self.model.temporal_extractor.parameters(), 'lr': base_lr * 1.5},
            {'params': self.model.prediction_layer.parameters(), 'lr': base_lr * 2.0}
        ]
        
        if optimizer_type.lower() == 'adamw':
            return optim.AdamW(param_groups, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'adam':
            return optim.Adam(param_groups, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'sgd':
            return optim.SGD(param_groups, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
    def _setup_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler"""
        scheduler_config = self.config.get('scheduler', {})
        if not scheduler_config:
            return None
        
        scheduler_type = scheduler_config.get('type', 'cosine')
        
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config['training']['epochs'],
                eta_min=scheduler_config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 10),
                min_lr=scheduler_config.get('min_lr', 1e-6)
            )
        else:
            return None
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {'total': 0.0, 'case': 0.0, 'r0': 0.0, 'risk': 0.0}
        epoch_metrics = {'accuracy': 0.0, 'mse': 0.0, 'auc': 0.0}
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            node_features = batch['node_features'].to(self.device)
            adjacency_matrix = batch['adjacency_matrix'].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch['targets'].items()}

            self.optimizer.zero_grad()
            predictions = self.model(node_features, adjacency_matrix)
            losses = self._calculate_detailed_loss(predictions, targets)
            total_loss = losses['total']
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            for key, value in losses.items():
                epoch_losses[key] += value.item()

            metrics = self._calculate_metrics(predictions, targets)
            for key, value in metrics.items():
                epoch_metrics[key] += value

            progress_bar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'Acc': f"{metrics['accuracy']:.3f}"
            })

        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return {**epoch_losses, **epoch_metrics}
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        epoch_losses = {'total': 0.0, 'case': 0.0, 'r0': 0.0, 'risk': 0.0}
        epoch_metrics = {'accuracy': 0.0, 'mse': 0.0, 'auc': 0.0}
        num_batches = len(val_loader)
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
            
            for batch_idx, batch in enumerate(progress_bar):
                node_features = batch['node_features'].to(self.device)
                adjacency_matrix = batch['adjacency_matrix'].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch['targets'].items()}

                predictions = self.model(node_features, adjacency_matrix)
                losses = self._calculate_detailed_loss(predictions, targets)
                for key, value in losses.items():
                    epoch_losses[key] += value.item()
                metrics = self._calculate_metrics(predictions, targets)
                for key, value in metrics.items():
                    epoch_metrics[key] += value
          
                all_predictions.append({k: v.cpu() for k, v in predictions.items()})
                all_targets.append({k: v.cpu() for k, v in targets.items()})
                progress_bar.set_postfix({
                    'Val Loss': f"{losses['total'].item():.4f}",
                    'Val Acc': f"{metrics['accuracy']:.3f}"
                })
        
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        epi_metrics = calculate_epidemiological_metrics(all_predictions, all_targets)
        
        return {**epoch_losses, **epoch_metrics, **epi_metrics}
    
    def _calculate_detailed_loss(self, predictions: Dict[str, torch.Tensor], 
                                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Calculate detailed loss components"""
        losses = {}
        total_loss = 0.0
        if 'case_predictions' in predictions and 'case_targets' in targets:
            case_loss = F.cross_entropy(predictions['case_predictions'], targets['case_targets'])
            losses['case'] = case_loss
            total_loss += self.loss_weights['case'] * case_loss

        if 'r0_estimates' in predictions and 'r0_targets' in targets:
            r0_loss = F.mse_loss(predictions['r0_estimates'], targets['r0_targets'])
            losses['r0'] = r0_loss
            total_loss += self.loss_weights['r0'] * r0_loss

        if 'risk_maps' in predictions and 'risk_targets' in targets:
            risk_loss = F.binary_cross_entropy(predictions['risk_maps'], targets['risk_targets'])
            losses['risk'] = risk_loss
            total_loss += self.loss_weights['risk'] * risk_loss

        reg_loss = self._calculate_regularization_loss()
        total_loss += reg_loss
        
        losses['total'] = total_loss
        return losses
    
    def _calculate_regularization_loss(self) -> torch.Tensor:
        """Calculate regularization losses (L1, L2, graph Laplacian)"""
        reg_loss = 0.0
        l1_lambda = self.config.get('l1_lambda', 1e-5)
        for name, param in self.model.named_parameters():
            if 'attention' in name and param.dim() > 1:
                reg_loss += l1_lambda * torch.norm(param, 1)
        
        return reg_loss
    
    def _calculate_metrics(self, predictions: Dict[str, torch.Tensor], 
                          targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        metrics = {}
        if 'case_predictions' in predictions and 'case_targets' in targets:
            pred_classes = torch.argmax(predictions['case_predictions'], dim=-1)
            accuracy = (pred_classes == targets['case_targets']).float().mean().item()
            metrics['accuracy'] = accuracy
        if 'r0_estimates' in predictions and 'r0_targets' in targets:
            mse = F.mse_loss(predictions['r0_estimates'], targets['r0_targets']).item()
            metrics['mse'] = mse

        if 'risk_maps' in predictions and 'risk_targets' in targets:
            try:
                risk_pred = predictions['risk_maps'].cpu().numpy().flatten()
                risk_true = targets['risk_targets'].cpu().numpy().flatten()
                auc = roc_auc_score(risk_true, risk_pred)
                metrics['auc'] = auc
            except ValueError:
                metrics['auc'] = 0.0
        
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List]:
        """Main training loop"""
        num_epochs = self.config['training']['epochs']
        save_dir = self.config.get('save_dir', './checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch+1}/{num_epochs}")
            train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_metrics['total'])
            self.metrics_history['train'].append(train_metrics)
            val_metrics = self.validate_epoch(val_loader)
            self.val_losses.append(val_metrics['total'])
            self.metrics_history['val'].append(val_metrics)
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['total'])
                else:
                    self.scheduler.step()
            self._log_metrics(epoch, train_metrics, val_metrics)
            if val_metrics['total'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total']
                self.early_stop_counter = 0
                
                checkpoint_path = os.path.join(save_dir, 'best_model.pth')
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    loss=val_metrics['total'],
                    path=checkpoint_path
                )
                self.logger.info(f"Saved best model with validation loss: {val_metrics['total']:.4f}")
            else:
                self.early_stop_counter += 1
            if self.early_stop_counter >= self.patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    loss=val_metrics['total'],
                    path=checkpoint_path
                )
        
        self.logger.info("Training completed")
        return self.metrics_history
    
    def _log_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log metrics to console and wandb"""
        self.logger.info(f"Train Loss: {train_metrics['total']:.4f}, "
                        f"Train Acc: {train_metrics['accuracy']:.3f}")
        self.logger.info(f"Val Loss: {val_metrics['total']:.4f}, "
                        f"Val Acc: {val_metrics['accuracy']:.3f}")
        
        if 'r0_mae' in val_metrics:
            self.logger.info(f"R0 MAE: {val_metrics['r0_mae']:.3f}, "
                           f"Attack Rate Error: {val_metrics.get('attack_rate_error', 0):.3f}")

        if self.use_wandb:
            log_dict = {
                'epoch': epoch,
                'train/loss': train_metrics['total'],
                'train/accuracy': train_metrics['accuracy'],
                'val/loss': val_metrics['total'],
                'val/accuracy': val_metrics['accuracy'],
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            
            for key in ['case', 'r0', 'risk']:
                if key in train_metrics:
                    log_dict[f'train/{key}_loss'] = train_metrics[key]
                if key in val_metrics:
                    log_dict[f'val/{key}_loss'] = val_metrics[key]

            for key in ['r0_mae', 'attack_rate_error', 'peak_time_error']:
                if key in val_metrics:
                    log_dict[f'epi/{key}'] = val_metrics[key]
            
            wandb.log(log_dict)
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
 
        train_acc = [m['accuracy'] for m in self.metrics_history['train']]
        val_acc = [m['accuracy'] for m in self.metrics_history['val']]
        axes[0, 1].plot(epochs, train_acc, 'b-', label='Train Accuracy')
        axes[0, 1].plot(epochs, val_acc, 'r-', label='Val Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # R0 MSE curves
        if 'mse' in self.metrics_history['train'][0]:
            train_mse = [m['mse'] for m in self.metrics_history['train']]
            val_mse = [m['mse'] for m in self.metrics_history['val']]
            axes[1, 0].plot(epochs, train_mse, 'b-', label='Train R0 MSE')
            axes[1, 0].plot(epochs, val_mse, 'r-', label='Val R0 MSE')
            axes[1, 0].set_title('R0 Estimation MSE')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('MSE')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
  
        if 'auc' in self.metrics_history['train'][0]:
            train_auc = [m['auc'] for m in self.metrics_history['train']]
            val_auc = [m['auc'] for m in self.metrics_history['val']]
            axes[1, 1].plot(epochs, train_auc, 'b-', label='Train AUC')
            axes[1, 1].plot(epochs, val_auc, 'r-', label='Val AUC')
            axes[1, 1].set_title('Risk Mapping AUC')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('AUC')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training curves saved to {save_path}")
        
        plt.show()

def main():
    """Main training script"""
    # Configuration
    config = {
        'model': {
            'input_dim': 8,
            'embedding_dim': 32,
            'spatial_dim': 64,
            'temporal_dim': 128,
            'output_dim': 5,
            'num_locations': 20,
            'num_gcn_layers': 3,
            'num_lstm_layers': 2,
            'dropout': 0.1
        },
        'training': {
            'epochs': 100,
            'batch_size': 32,
            'validation_split': 0.2
        },
        'optimizer': {
            'type': 'adamw',
            'lr': 1e-3,
            'weight_decay': 1e-4
        },
        'scheduler': {
            'type': 'cosine',
            'min_lr': 1e-6
        },
        'loss_weights': {
            'case': 1.0,
            'r0': 0.5,
            'risk': 0.3
        },
        'patience': 20,
        'save_dir': './checkpoints',
        'l1_lambda': 1e-5,
        'log_level': 'INFO'
    }

    train_dataset = EpidemicDataset('train', config)
    val_dataset = EpidemicDataset('val', config)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        num_workers=4
    )

    trainer = DeepEVDTrainer(config, use_wandb=False)
    history = trainer.train(train_loader, val_loader)
    trainer.plot_training_curves('./training_curves.png')
    torch.save(trainer.model.state_dict(), './final_model.pth')
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()