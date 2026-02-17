import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm
import os

class MultimodalTrainer:
    def __init__(self, 
                 model, 
                 train_dataset, 
                 val_dataset, 
                 config):
        self.config = config
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
            mixed_precision=config.get('mixed_precision', 'bf16')
        )
        
        self.model = model
        self.train_dataloader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True,
            num_workers=config.get('num_workers', 4)
        )
        self.val_dataloader = DataLoader(
            val_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False,
            num_workers=config.get('num_workers', 4)
        )
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config['num_epochs'] * len(self.train_dataloader)
        )
        
        # Prepare for distributed training
        self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.lr_scheduler
        )

    def train(self):
        for epoch in range(self.config['num_epochs']):
            self.model.train()
            total_loss = 0
            
            progress_bar = tqdm(self.train_dataloader, disable=not self.accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")
            
            for step, batch in enumerate(progress_bar):
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        images=batch.get('images'),
                        targets=batch['targets'],
                        function_targets={
                            'call': batch['call_label'],
                            'func': batch['func_label']
                        }
                    )
                    
                    loss = outputs['loss']
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.get('max_grad_norm', 1.0))
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    
                    total_loss += loss.item()
                    progress_bar.set_postfix(loss=loss.item())
            
            avg_loss = total_loss / len(self.train_dataloader)
            self.accelerator.print(f"Epoch {epoch} average loss: {avg_loss:.4f}")
            
            # Validation
            self.validate(epoch)
            
            # Save checkpoint
            if self.accelerator.is_local_main_process:
                self.save_checkpoint(epoch)

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        for batch in self.val_dataloader:
            outputs = self.model(
                input_ids=batch['input_ids'],
                images=batch.get('images'),
                targets=batch['targets'],
                function_targets={
                    'call': batch['call_label'],
                    'func': batch['func_label']
                }
            )
            val_loss += outputs['loss'].item()
            
        avg_val_loss = val_loss / len(self.val_dataloader)
        self.accelerator.print(f"Epoch {epoch} validation loss: {avg_val_loss:.4f}")

    def save_checkpoint(self, epoch):
        output_dir = os.path.join(self.config['output_dir'], f"checkpoint-epoch-{epoch}")
        os.makedirs(output_dir, exist_ok=True)
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        torch.save(unwrapped_model.state_dict(), os.path.join(output_dir, "model.pt"))
        self.accelerator.print(f"Saved checkpoint to {output_dir}")
