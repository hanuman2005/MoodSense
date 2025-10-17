"""
Model training utilities for MoodSense
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoConfig
import torchvision.models as models
import numpy as np
import cv2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json

from utils import EmotionUtils, ModelUtils

logger = logging.getLogger(__name__)

class TextEmotionModel(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased", num_emotions: int = 13, dropout: float = 0.3):
        super(TextEmotionModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_emotions)

        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        output = self.dropout(pooled_output)
        return self.classifier(output)
    
    def unfreeze_bert(self):
        """Unfreeze BERT parameters for fine-tuning"""
        for param in self.bert.parameters():
            param.requires_grad = True

class ImageEmotionModel(nn.Module):
    """CNN-based image emotion recognition model"""
    
    def __init__(self, num_emotions: int = 7, pretrained: bool = True):
        super(ImageEmotionModel, self).__init__()
        
        # Use MobileNetV2 as backbone
        self.backbone = models.mobilenet_v2(pretrained=pretrained)
        
        # Modify first layer to accept grayscale input
        self.backbone.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.backbone.last_channel, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_emotions)
        )
    
    def forward(self, x):
        return self.backbone(x)

class ModelTrainer:
    """Main model training class"""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device if device else ModelUtils.get_device()
        self.emotion_labels = EmotionUtils.EMOTION_LABELS
        
    def train_text_model(self, train_loader: DataLoader, val_loader: DataLoader,
                     model_save_path: str, num_epochs: int = 10,
                     learning_rate: float = 2e-5, num_classes: int = 13,
                     train_labels: np.ndarray = None) -> Dict:

        """Train text emotion recognition model"""
        
        # Initialize model
        model = TextEmotionModel(num_emotions=num_classes)
        if num_epochs >= 3:
            model.unfreeze_bert()
        model = model.to(self.device)
        from sklearn.utils.class_weight import compute_class_weight

        class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=weights)

        # Loss and optimizer
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
        
        # Training history
        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
        
        best_val_acc = 0.0
        patience = 3
        no_improve_epochs = 0

        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Training phase
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            
            train_pbar = tqdm(train_loader, desc="Training")
            for batch in train_pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                train_pbar.set_postfix({'loss': loss.item()})
            
            # Validation phase
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Calculate metrics
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # Update history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            logger.info(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            logger.info(f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve_epochs = 0  # ✅ Reset counter
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'emotion_labels': self.emotion_labels,
                'num_classes': num_classes
                }, model_save_path)
                logger.info(f"Best model saved with validation accuracy: {val_acc:.2f}%")
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    logger.info("⏹️ Early stopping triggered")
                    break

        
        return history
    
    def train_image_model(self, train_loader: DataLoader, val_loader: DataLoader,
                         model_save_path: str, num_epochs: int = 20,
                         learning_rate: float = 0.001) -> Dict:
        """Train image emotion recognition model"""
        
        # Initialize model
        model = ImageEmotionModel(num_emotions=len(self.emotion_labels))
        model = model.to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        # Training history
        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Training phase
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            
            train_pbar = tqdm(train_loader, desc="Training")
            for images, labels in train_pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                train_pbar.set_postfix({'loss': loss.item()})
            
            # Validation phase
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc="Validation"):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Calculate metrics
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # Update history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            scheduler.step()
            
            logger.info(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            logger.info(f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'emotion_labels': self.emotion_labels
                }, model_save_path)
                logger.info(f"Best model saved with validation accuracy: {val_acc:.2f}%")
        
        return history
    
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader, 
                      model_type: str = "text") -> Dict:
        """Evaluate trained model"""
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                if model_type == "text":
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    outputs = model(input_ids, attention_mask)
                else:  # image
                    images, labels = batch
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = model(images)
                
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        report = classification_report(all_labels, all_predictions, 
                                     target_names=self.emotion_labels, output_dict=True)
        cm = confusion_matrix(all_labels, all_predictions)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'true_labels': all_labels
        }
    
    def plot_training_history(self, history: Dict, save_path: str = None):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        axes[0].plot(history['train_loss'], label='Training Loss')
        axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        
        # Plot accuracy
        axes[1].plot(history['train_acc'], label='Training Accuracy')
        axes[1].plot(history['val_acc'], label='Validation Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str = None):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.emotion_labels,
                   yticklabels=self.emotion_labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Confusion matrix plot saved to {save_path}")
        
        plt.show()
    
    def load_trained_model(self, model_path: str, model_type: str = "text"):
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            num_classes = checkpoint.get("num_classes", len(self.emotion_labels))  # fallback to 7 if missing

            if model_type == "text":
                model = TextEmotionModel(num_emotions=num_classes)
            elif model_type == "image":
                model = ImageEmotionModel(num_emotions=num_classes)
            else:
                raise ValueError("Unsupported model type")

            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(self.device)
            model.eval()
            return model

        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            return None


    
    def save_training_config(self, config: Dict, save_path: str):
        """Save training configuration"""
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(config, f, indent=4)
            logger.info(f"Training config saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving config to {save_path}: {e}")
    
    def predict_single_text(self, model: TextEmotionModel, text: str, 
                           tokenizer_name: str = "distilbert-base-uncased") -> Dict[str, float]:
        """Predict emotion for a single text"""
        from transformers import AutoTokenizer
        
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Preprocess text
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
        # Convert to emotion scores
        emotion_scores = {}
        for i, emotion in enumerate(self.emotion_labels):
            emotion_scores[emotion] = float(probabilities[0][i])
        
        return emotion_scores
    
    def predict_single_image(self, model: ImageEmotionModel, image: np.ndarray) -> Dict[str, float]:
        """Predict emotion for a single image"""
        model.eval()
        
        # Preprocess image
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        image = cv2.resize(image, (48, 48))
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(np.expand_dims(image, axis=0), axis=0)
        image = torch.tensor(image).to(self.device)
        
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Convert to emotion scores
        emotion_scores = {}
        for i, emotion in enumerate(self.emotion_labels):
            emotion_scores[emotion] = float(probabilities[0][i])
        
        return emotion_scores