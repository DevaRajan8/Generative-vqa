import matplotlib.pyplot as plt
import json
import os
class TrainingVisualizer:

    def __init__(self, save_dir='training_logs'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
    
    def update(self, train_loss, train_acc, val_loss, val_acc, lr):
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['learning_rates'].append(lr)
    
    def plot(self):

        epochs = range(1, len(self.history['train_loss']) + 1)
        

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        

        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        

        axes[1].plot(epochs, self.history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        axes[1].plot(epochs, self.history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        

        axes[2].plot(epochs, self.history['learning_rates'], 'g-', linewidth=2)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].set_yscale('log') 
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/training_progress.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training plot saved to {self.save_dir}/training_progress.png")
    
    def save_history(self):

        with open(f'{self.save_dir}/training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {self.save_dir}/training_history.json")