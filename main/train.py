import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from data_utils import get_dataloaders, get_spectrogram_dataloaders, Datasets
from model_utils import get_model, ModelType
import argparse
import logging


MODELS_DIR = "../models"

def get_inverse_class_frequencies(dataloader):
    """
    Computes inverse class frequencies from a DataLoader and rescales them 
    so the smallest non-zero value is 1.
    
    Args:
        dataloader: DataLoader yielding (X, y) batches
    Returns:
        num_classes (int): number of detected classes
        alpha_list (list): rescaled inverse frequencies in class index order
    """
    all_labels = []

    for _, labels in dataloader:
        all_labels.append(labels.view(-1))  # flatten batch

    all_labels = torch.cat(all_labels)
    num_classes = int(all_labels.max().item()) + 1  # auto-detect

    counts = torch.bincount(all_labels, minlength=num_classes).float()
    inverse_freq = torch.where(counts > 0, 1.0 / counts, torch.zeros_like(counts))

    # Rescale so smallest non-zero becomes 1
    min_val = inverse_freq[inverse_freq > 0].min()
    inverse_freq = inverse_freq / min_val

    return torch.tensor(inverse_freq.tolist(), dtype=torch.float)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss wrapper around CrossEntropyLoss.

        Args:
            alpha (None, list, tensor): Per-class weighting factor.
                                         - None: no weighting
                                         - list/tensor: per-class weights
            gamma (float): Focusing parameter.
            reduction (str): 'none' | 'mean' | 'sum'
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is None:
            ce_alpha = None
        else:
            if isinstance(alpha, (list, tuple)):
                ce_alpha = torch.tensor(alpha, dtype=torch.float)
            elif isinstance(alpha, torch.Tensor):
                ce_alpha = alpha.float()
            else:
                raise TypeError("alpha must be None, list, tuple, or torch.Tensor.")

        # Register buffer so it moves automatically to GPU with the model
        self.register_buffer("alpha", ce_alpha if alpha is not None else torch.tensor([]))

    def forward(self, inputs, targets):
        # If alpha is empty, use None
        if self.alpha.numel() > 0:
            ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha.to(inputs.device), reduction='none')
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        pt = torch.exp(-ce_loss)  # equivalent to probs.gather(...)
        focal_factor = (1 - pt) ** self.gamma
        loss = focal_factor * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
        
class CombinedCrossEntropyLoss(nn.Module):
    def __init__(self, model=None, lambda1=0.5):
        super(CombinedCrossEntropyLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.lambda1 = lambda1

    def forward(self, logits, targets):
        logits_ad, logits_lang = logits
        targets_ad, targets_lang = targets

        loss_ad = self.ce_loss(logits_ad, targets_ad)
        loss_lang = self.ce_loss(logits_lang, targets_lang)

        total_loss = loss_ad + (self.lambda1 * loss_lang)
        return total_loss


class TrainHandler():
    def __init__(self, model_type, train_loader, val_loader, test_loader, device, lr, weight_decay, criterion, lang_awareness, continue_training=False):
        self.model_type = model_type

        if model_type not in (model.value for model in ModelType):
            logging.error(f"Model type {model_type} is not valid. Choose from {list(ModelType.__members__.keys())}")
            raise ValueError(f"Model type {model_type} is not valid. Choose from {list(ModelType.__members__.keys())}")

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device

        self.criterion = criterion

        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

        self.best_epoch_val_loss = float('inf')
        self.best_epoch = 0
        self.last_epoch_counter = 0
        self.last_epoch = 0

        self.checkpoint_dir = CHECKPOINT_DIR
        self.load_model_n_stats(continue_training=continue_training, lr=lr, weight_decay=weight_decay, lang_awareness=lang_awareness)

        self.lang_aware = lang_awareness

        self.best_model = None
        self.best_optimizer = None
        self.best_scheduler = None
        logging.info(f"TrainHandler initialized for model: {model_type} with language awareness: {lang_awareness}\n")

    def load_model_n_stats(self, continue_training, lr, weight_decay, lang_awareness):
        """
        Load the model and training statistics from the last checkpoint.
        """

        logging.info("Initializing model and optimizer...")

        if continue_training and os.path.exists(self.checkpoint_dir) and any(f.endswith('.pth') for f in os.listdir(self.checkpoint_dir)):
            logging.info(f"Loading model and training statistics from {self.checkpoint_dir}")
            checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            self.model = get_model(self.model_type, checkpoint_path, lang_awareness).to(self.device)
            optimizer = optim.Adam(self.model.parameters(), lr=3e-3, weight_decay=1e-5)
            optimizer.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, "optimizer.pth")))
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)
            scheduler.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, "scheduler.pth")))

            prev_history = torch.load(os.path.join(self.checkpoint_dir, "history.pth"))
            self.history['train_loss'] = prev_history['train_loss']
            self.history['train_acc'] = prev_history['train_acc']
            self.history['val_loss'] = prev_history['val_loss']
            self.history['val_acc'] = prev_history['val_acc']
            self.best_epoch_val_loss = prev_history['best_epoch_val_loss']
            self.best_epoch = prev_history.get('best_epoch', 0)
            self.last_epoch_counter = prev_history['last_epoch_counter']
            self.last_epoch = len(self.history['train_loss'])

            logging.info(f"Model and optimizer loaded from {self.checkpoint_dir}")
            logging.info(f"Last epoch: {self.last_epoch} with best val loss: {self.best_epoch_val_loss:.4f} at epoch {self.best_epoch}\n")
        else:
            checkpoint_path = None
            self.model = get_model(self.model_type, checkpoint_path, lang_awareness).to(self.device)
            logging.info("No checkpoint found. Starting training from scratch.")
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)
            logging.info("Model and optimizer initialized.\n")

        self.optimizer = optimizer
        self.scheduler = scheduler

    def save_model_n_stats(self, current_epoch):
        """
        Save the model and training statistics.
        """
        logging.info(f"Saving model and training statistics to {self.checkpoint_dir}")
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, "model.pth"))
        torch.save(self.optimizer.state_dict(), os.path.join(self.checkpoint_dir, "optimizer.pth"))
        torch.save(self.scheduler.state_dict(), os.path.join(self.checkpoint_dir, "scheduler.pth"))

        torch.save(self.best_model, os.path.join(self.checkpoint_dir, "best_model.pth")) if self.best_model else None
        torch.save(self.best_optimizer, os.path.join(self.checkpoint_dir, "best_optimizer.pth")) if self.best_optimizer else None
        torch.save(self.best_scheduler, os.path.join(self.checkpoint_dir, "best_scheduler.pth")) if self.best_scheduler else None

        history = {
            'train_loss': self.history['train_loss'],
            'train_acc': self.history['train_acc'],
            'val_loss': self.history['val_loss'],
            'val_acc': self.history['val_acc'],
            'best_epoch_val_loss': self.best_epoch_val_loss,
            'best_epoch': self.best_epoch,
            'last_epoch_counter': self.last_epoch_counter
        }
        torch.save(history, os.path.join(self.checkpoint_dir, "history.pth"))
        logging.info(f"Model checkpoint saved at epoch {current_epoch} with validation loss: {self.best_epoch_val_loss:.4f}\n")
    
    def train_epoch(self, epoch_num):
        """
        Train one epoch of the model.
        """
        logging.info(f"Starting training for epoch {epoch_num}")
        self.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_bar = tqdm(self.train_loader, desc="Training")
        for inputs, labels in train_bar:
            inputs = inputs.to(self.device)
            if self.lang_aware:
                labels = (labels[0].to(self.device), labels[1].to(self.device))
            else:
                labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

            if self.lang_aware:
                _, predicted = torch.max(outputs[0].data, 1)
                train_total += labels[0].size(0)
                train_correct += (predicted == labels[0]).sum().item()
            else:
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            train_bar.set_postfix(loss=train_loss/len(train_bar),
                                    acc=100.*train_correct/train_total)
        epoch_train_loss = train_loss / len(self.train_loader)
        epoch_train_acc = 100. * train_correct / train_total
        self.history['train_loss'].append(epoch_train_loss)
        self.history['train_acc'].append(epoch_train_acc)

        logging.info(f"Epoch {epoch_num} completed - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%")
        return epoch_train_loss, epoch_train_acc

    def validate_epoch(self, epoch_num):
        """
        Validate on the validation set.
        """
        logging.info(f"Starting validation for epoch {epoch_num}")
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            val_bar = tqdm(self.val_loader, desc="Validation")
            for inputs, labels in val_bar:
                inputs = inputs.to(self.device)
                if self.lang_aware:
                    labels = (labels[0].to(self.device), labels[1].to(self.device))
                else:
                    labels = labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                if self.lang_aware:
                    _, predicted = torch.max(outputs[0].data, 1)
                    val_total += labels[0].size(0)
                    val_correct += (predicted == labels[0]).sum().item()
                else:
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                val_bar.set_postfix(loss=val_loss/len(val_bar),
                                    acc=100.*val_correct/val_total)
        epoch_val_loss = val_loss / len(self.val_loader)
        epoch_val_acc = 100. * val_correct / val_total
        self.history['val_loss'].append(epoch_val_loss)
        self.history['val_acc'].append(epoch_val_acc)

        logging.info(f"Epoch {epoch_num} validation - Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%\n")
        return epoch_val_loss, epoch_val_acc

    def train(self, num_epochs=50, patience=7):
        """
        Train the model for specified number of epochs with early stopping.
        """
        logging.info(f"Starting training process for {num_epochs} epochs with patience {patience}\n")
        for epoch in range(self.last_epoch + 1, self.last_epoch + num_epochs + 1):
            current_epoch = epoch
            logging.info(f"=== Starting Epoch {epoch}/{self.last_epoch + num_epochs} ===")
            
            epoch_train_loss, epoch_train_acc = self.train_epoch(epoch)
            epoch_val_loss, epoch_val_acc = self.validate_epoch(epoch)

            prev_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(epoch_val_loss)
            curr_lr = self.optimizer.param_groups[0]['lr']
            
            if curr_lr != prev_lr:
                logging.info(f"Learning rate adjusted from {prev_lr} to {curr_lr}")

            if epoch_val_loss < self.best_epoch_val_loss:
                self.best_epoch_val_loss = epoch_val_loss
                self.best_epoch = epoch
                self.best_model = self.model.state_dict()
                self.best_optimizer = self.optimizer.state_dict()
                self.best_scheduler = self.scheduler.state_dict()
                self.last_epoch_counter = 0
                self.save_model_n_stats(current_epoch)
                logging.info(f"New best model saved at epoch {epoch} with validation loss: {epoch_val_loss:.4f}")
            else:
                self.last_epoch_counter += 1
                logging.info(f"No improvement for {self.last_epoch_counter} epochs. Best val loss: {self.best_epoch_val_loss:.4f} at epoch {self.best_epoch}")
                if self.last_epoch_counter >= patience:
                    logging.info(f"Early stopping triggered after {epoch} epochs\n")
                    break
            
            logging.info(f"=== Completed Epoch {epoch}/{self.last_epoch + num_epochs} ===\n")

        logging.info(f"Training completed. Best validation loss: {self.best_epoch_val_loss:.4f} at epoch {self.best_epoch}\n")
        return

    def evaluate(self):
        """
        Evaluate the model on the test set.
        """
        logging.info("Starting model evaluation on test set")
        self.model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc="Evaluating"):
                inputs = inputs.to(self.device)
                if self.lang_aware:
                    labels = (labels[0].to(self.device), labels[1].to(self.device))
                else:
                    labels = labels.to(self.device)
                outputs = self.model(inputs)

                if self.lang_aware:
                    _, predicted = torch.max(outputs[0].data, 1)
                    all_labels.extend(labels[0].cpu().numpy())
                    all_preds.extend(predicted.cpu().numpy())
                else:
                    _, predicted = torch.max(outputs.data, 1)
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(predicted.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)

        logging.info(f"Test Evaluation Results:")
        logging.info(f"Test Accuracy: {acc*100:.2f}%")
        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        logging.info(f"F1 Score: {f1:.4f}")

        # save the confusion matrix
        logging.info("Generating confusion matrix")
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Non-AD', 'AD'], 
                    yticklabels=['Non-AD', 'AD'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(f"{self.checkpoint_dir}/confusion_matrix.png")
        logging.info(f"Confusion matrix saved to {self.checkpoint_dir}/confusion_matrix.png")
        plt.close()

        return 

    def plot_training_history(self):
        """
        Plot and save the training and validation loss and accuracy.
        """
        logging.info("Generating training history plots")
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train')
        plt.plot(self.history['val_loss'], label='Validation')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Train')
        plt.plot(self.history['val_acc'], label='Validation')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.checkpoint_dir}/training_history.png")
        logging.info(f"Training history plots saved to {self.checkpoint_dir}/training_history.png")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model for AD detection")
    parser.add_argument('--model', type=str, choices=[model.value for model in ModelType], default="wavenet", help="Model architecture to train")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs to train on")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--learning_rate', type=float, default=3e-3, help="Learning rate for optimizer - only when not continuing training")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="Weight decay for optimizer - only when not continuing training")
    parser.add_argument('--patience', type=int, default=3, help="Patience for early stopping")
    parser.add_argument('--continue_training', type=int, choices=[0, 1], default=1, help="1 if continuing training, 0 if starting from scratch")
    parser.add_argument('--dataset', type=str, choices=[dataset.value for dataset in Datasets], default="adress", help="Dataset to train the model on")
    parser.add_argument('--lang_awareness', type=int, choices=[0, 1], default=1, help="1 for language aware training, 0 for only dimentia training")
    args = parser.parse_args()

    MODEL_NAME = args.model
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    WEIGHT_DECAY = args.weight_decay
    PATIENCE = args.patience
    CONTINUE_TRAINING = bool(args.continue_training)
    DATASET = args.dataset
    LANG_AWARENESS = bool(args.lang_awareness)
    if DATASET != Datasets.ALL.value:
        LANG_AWARENESS = False

    
    CHECKPOINT_DIR = os.path.join(MODELS_DIR, f"{MODEL_NAME}", f"{DATASET}")
    if LANG_AWARENESS:
        CHECKPOINT_DIR += "_lang_aware"

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    logging.basicConfig(
        filename=os.path.join(CHECKPOINT_DIR, 'training.log'),
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("=" * 80)
    logging.info(f"NEW TRAINING RUN: {MODEL_NAME} on {DATASET} dataset with language awareness: {LANG_AWARENESS}")
    logging.info(f"Parameters: Epochs={EPOCHS}, Batch Size={BATCH_SIZE}, LR={LEARNING_RATE}, Weight Decay={WEIGHT_DECAY}")
    logging.info(f"Continue Training: {CONTINUE_TRAINING}, Patience: {PATIENCE}")
    logging.info("=" * 80)

    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}\n")

    logging.info("Loading dataset...")
    if MODEL_NAME == ModelType.SPEC_CNN.value:
        dataloaders = get_spectrogram_dataloaders(dataset_name=DATASET, batch_size=BATCH_SIZE, lang_aware=LANG_AWARENESS)
    else:
        dataloaders = get_dataloaders(dataset_name=DATASET, batch_size=BATCH_SIZE, lang_aware=LANG_AWARENESS)
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']
    logging.info(f"Dataset loaded - Train set: {len(train_loader.dataset)}, Validation set: {len(val_loader.dataset)}, Test set: {len(test_loader.dataset)}\n")

    
    if LANG_AWARENESS:
        criterion = CombinedCrossEntropyLoss()
        logging.info("Using CombinedCrossEntropyLoss for language-aware training")
    else:
        # alpha_cls = get_inverse_class_frequencies(train_loader)
        # criterion = FocalLoss(alpha=alpha_cls)
        # logging.info("Using Focal Loss")
        criterion = nn.CrossEntropyLoss()
        logging.info("Using standard CrossEntropyLoss")

    train_handler = TrainHandler(
        model_type=MODEL_NAME,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        lr = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY,
        criterion = criterion,
        lang_awareness=LANG_AWARENESS,
        continue_training=CONTINUE_TRAINING,
    )

    logging.info("Starting model training process")
    train_handler.train(num_epochs=EPOCHS, patience=PATIENCE)
    
    logging.info("Training completed. Starting model evaluation")
    train_handler.evaluate()
    
    train_handler.plot_training_history()
    
    logging.info("=" * 80)
    logging.info("TRAINING SUMMARY:")
    logging.info(f"Model: {MODEL_NAME} on {DATASET} dataset")
    logging.info(f"Best validation loss: {train_handler.best_epoch_val_loss:.4f} achieved at epoch {train_handler.best_epoch}")
    logging.info(f"Final training accuracy: {train_handler.history['train_acc'][-1]:.2f}%")
    logging.info(f"Final validation accuracy: {train_handler.history['val_acc'][-1]:.2f}%")
    logging.info(f"Model saved at: {CHECKPOINT_DIR}")
    logging.info("Training completed successfully")
    logging.info("=" * 80)
    
