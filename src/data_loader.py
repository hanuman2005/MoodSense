"""
Data loading and preprocessing utilities for MoodSense
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import torch
from transformers import AutoTokenizer
import logging
from typing import List, Tuple, Dict, Optional
import glob
import json

from utils import EmotionUtils, ImageUtils, TextUtils, DataUtils

logger = logging.getLogger(__name__)

class TextEmotionDataset(Dataset):
    """Dataset class for text emotion recognition"""
    
    def __init__(self, texts: List[str], labels: List[str], tokenizer_name: str = "distilbert-base-uncased",
                 max_length: int = 512, label_encoder: Optional[LabelEncoder] = None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        # Use the provided LabelEncoder or fit a new one
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.encoded_labels = self.label_encoder.fit_transform(labels)
        else:
            self.label_encoder = label_encoder
            self.encoded_labels = self.label_encoder.transform(labels)
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.encoded_labels[idx]
        
        clean_text = TextUtils.clean_text(text)
        encoding = self.tokenizer(
            clean_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class ImageEmotionDataset(Dataset):
    """Dataset class for image emotion recognition"""
    
    def __init__(self, image_paths: List[str], labels: List[str], transform=None, label_encoder: Optional[LabelEncoder] = None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
        # Use provided LabelEncoder or fit a new one
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.encoded_labels = self.label_encoder.fit_transform(labels)
        else:
            self.label_encoder = label_encoder
            self.encoded_labels = self.label_encoder.transform(labels)
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.encoded_labels[idx]
        
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                image = Image.open(image_path).convert('L')
                image = np.array(image)
            
            image = cv2.resize(image, (48, 48))
            image = image.astype(np.float32) / 255.0
            image = np.expand_dims(image, axis=0)
            
            if self.transform:
                image = self.transform(image)
            
            return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            zero_image = np.zeros((1, 48, 48), dtype=np.float32)
            return torch.tensor(zero_image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class DataLoader:
    """Main data loading and preprocessing class"""
    
    def __init__(self, data_dir: str = "data/"):
        self.data_dir = data_dir
        self.emotion_labels = EmotionUtils.EMOTION_LABELS
        
    def load_text_emotion_data(self, dataset_path: str) -> Tuple[List[str], List[str]]:
        texts, labels = [], []
        try:
            if dataset_path.endswith('.csv'):
                df = pd.read_csv(dataset_path)
                text_cols = ['text', 'Text', 'sentence', 'Sentence', 'content', 'Content']
                label_cols = ['emotion', 'Emotion', 'label', 'Label', 'sentiment', 'Sentiment']
                text_col = next((col for col in text_cols if col in df.columns), None)
                label_col = next((col for col in label_cols if col in df.columns), None)
                if text_col and label_col:
                    texts = df[text_col].astype(str).tolist()
                    labels = df[label_col].astype(str).tolist()
                else:
                    logger.error(f"Could not find text and label columns in {dataset_path}")
            elif dataset_path.endswith('.json'):
                with open(dataset_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if 'text' in item and 'emotion' in item:
                            texts.append(str(item['text']))
                            labels.append(str(item['emotion']))
                        elif 'sentence' in item and 'label' in item:
                            texts.append(str(item['sentence']))
                            labels.append(str(item['label']))
            elif os.path.isdir(dataset_path):
                for emotion in self.emotion_labels:
                    emotion_dir = os.path.join(dataset_path, emotion)
                    if os.path.exists(emotion_dir):
                        for file_path in glob.glob(os.path.join(emotion_dir, "*.txt")):
                            with open(file_path, 'r', encoding='utf-8') as f:
                                text = f.read().strip()
                                texts.append(text)
                                labels.append(emotion)
        except Exception as e:
            logger.error(f"Error loading text emotion data from {dataset_path}: {e}")
        logger.info(f"Loaded {len(texts)} text samples with {len(set(labels))} unique emotions")
        return texts, labels

    # ----------------------------
    # Key fix: pass the same LabelEncoder to train and val datasets
    # ----------------------------
    def create_text_dataloaders(self, texts: List[str], labels: List[str], 
                                batch_size: int = 32, test_size: float = 0.2,
                                tokenizer_name: str = "distilbert-base-uncased") -> Tuple:
        # Fit LabelEncoder on full labels
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)
        
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        train_dataset = TextEmotionDataset(X_train, y_train, tokenizer_name, label_encoder=label_encoder)
        val_dataset = TextEmotionDataset(X_val, y_val, tokenizer_name, label_encoder=label_encoder)
        
        train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        return train_loader, val_loader, label_encoder

    # Similar fix for images
    def create_image_dataloaders(self, image_paths: List[str], labels: List[str],
                                 batch_size: int = 32, test_size: float = 0.2) -> Tuple:
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)
        
        X_train, X_val, y_train, y_val = train_test_split(
            image_paths, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        train_dataset = ImageEmotionDataset(X_train, y_train, label_encoder=label_encoder)
        val_dataset = ImageEmotionDataset(X_val, y_val, label_encoder=label_encoder)
        
        train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        return train_loader, val_loader, label_encoder

    # Rest of your original methods (unchanged)
    def load_fer2013_data(self, fer_path: str) -> Tuple[List[str], List[str]]:
        image_paths, labels = [], []
        try:
            if fer_path.endswith('.csv'):
                df = pd.read_csv(fer_path)
                if 'pixels' in df.columns and 'emotion' in df.columns:
                    temp_dir = os.path.join(self.data_dir, 'temp_fer2013')
                    os.makedirs(temp_dir, exist_ok=True)
                    emotion_map = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 
                                   4: 'sad', 5: 'surprise', 6: 'neutral'}
                    for idx, row in df.iterrows():
                        pixels = np.array(row['pixels'].split(), dtype=np.uint8)
                        image = pixels.reshape(48, 48)
                        emotion_label = emotion_map.get(row['emotion'], 'neutral')
                        img_path = os.path.join(temp_dir, f"{idx}_{emotion_label}.png")
                        cv2.imwrite(img_path, image)
                        image_paths.append(img_path)
                        labels.append(emotion_label)
            elif os.path.isdir(fer_path):
                for emotion in self.emotion_labels:
                    emotion_dir = os.path.join(fer_path, emotion)
                    if os.path.exists(emotion_dir):
                        for ext in ['*.png', '*.jpg', '*.jpeg']:
                            for img_path in glob.glob(os.path.join(emotion_dir, ext)):
                                image_paths.append(img_path)
                                labels.append(emotion)
        except Exception as e:
            logger.error(f"Error loading FER-2013 data from {fer_path}: {e}")
        logger.info(f"Loaded {len(image_paths)} image samples with {len(set(labels))} unique emotions")
        return image_paths, labels

    def load_spotify_data(self, spotify_path: str) -> pd.DataFrame:
        try:
            if spotify_path.endswith('.csv'):
                df = pd.read_csv(spotify_path)
                logger.info(f"Loaded Spotify dataset with {len(df)} songs")
                return df
            else:
                logger.error(f"Unsupported Spotify data format: {spotify_path}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading Spotify data from {spotify_path}: {e}")
            return pd.DataFrame()

    def create_food_mood_data(self) -> pd.DataFrame:
        food_path = os.path.join(self.data_dir, 'food_mood_mapping.csv')
        if os.path.exists(food_path):
            df = pd.read_csv(food_path)
            logger.info(f"Loaded existing food-mood mapping with {len(df)} items")
        else:
            df = DataUtils.create_food_mood_mapping()
            os.makedirs(os.path.dirname(food_path), exist_ok=True)
            df.to_csv(food_path, index=False)
            logger.info(f"Created new food-mood mapping with {len(df)} items")
        return df

    def get_data_statistics(self, labels: List[str]) -> Dict:
        from collections import Counter
        label_counts = Counter(labels)
        total_samples = len(labels)
        stats = {
            'total_samples': total_samples,
            'num_classes': len(set(labels)),
            'class_distribution': dict(label_counts),
            'class_percentages': {k: (v/total_samples)*100 for k, v in label_counts.items()}
        }
        return stats

    def validate_data_paths(self) -> Dict[str, bool]:
        paths_to_check = {
            'emotion_text_dataset': os.path.join(self.data_dir, 'emotion_text_dataset'),
            'fer2013': os.path.join(self.data_dir, 'fer2013'),
            'spotify_dataset': os.path.join(self.data_dir, 'spotify_dataset'),
            'food_mood_mapping': os.path.join(self.data_dir, 'food_mood_mapping.csv')
        }
        validation_results = {}
        for name, path in paths_to_check.items():
            validation_results[name] = os.path.exists(path)
            if validation_results[name]:
                logger.info(f"✓ Found: {name} at {path}")
            else:
                logger.warning(f"✗ Missing: {name} at {path}")
        return validation_results
