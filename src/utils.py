"""
Utility functions for MoodSense application
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import AutoTokenizer
import logging
from typing import List, Dict, Tuple, Optional, Union
import re
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionUtils:
    """Utility functions for emotion processing"""
    
    EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    EMOTION_COLORS = {
        'angry': '#FF4444',
        'disgust': '#44FF44', 
        'fear': '#4444FF',
        'happy': '#FFFF44',
        'neutral': '#888888',
        'sad': '#4488FF',
        'surprise': '#FF44FF'
    }
    
    @staticmethod
    def get_emotion_emoji(emotion: str) -> str:
        """Get emoji representation of emotion"""
        emoji_map = {
            'angry': '😠',
            'disgust': '🤢',
            'fear': '😨', 
            'happy': '😊',
            'neutral': '😐',
            'sad': '😢',
            'surprise': '😲'
        }
        return emoji_map.get(emotion.lower(), '😐')
    
    @staticmethod
    def normalize_emotion_scores(scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize emotion scores to sum to 1"""
        total = sum(scores.values())
        if total == 0:
            return {k: 1/len(scores) for k in scores}
        return {k: v/total for k, v in scores.items()}
    
    @staticmethod
    def get_dominant_emotion(scores: Dict[str, float]) -> str:
        """Get the emotion with highest score"""
        return max(scores.items(), key=lambda x: x[1])[0]
    
    @staticmethod
    def emotion_to_valence_arousal(emotion: str) -> Tuple[float, float]:
        """Convert emotion to valence-arousal coordinates"""
        mapping = {
            'happy': (0.8, 0.6),
            'surprise': (0.5, 0.8),
            'fear': (-0.6, 0.7),
            'angry': (-0.6, 0.8),
            'disgust': (-0.7, 0.4),
            'sad': (-0.7, -0.4),
            'neutral': (0.0, 0.0)
        }
        return mapping.get(emotion.lower(), (0.0, 0.0))

class ImageUtils:
    """Utility functions for image processing"""
    
    @staticmethod
    def preprocess_face_image(image: np.ndarray, target_size: Tuple[int, int] = (48, 48)) -> np.ndarray:
        """Preprocess face image for emotion recognition"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Resize image
        resized = cv2.resize(gray, target_size)
        
        # Normalize pixel values
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions
        preprocessed = np.expand_dims(np.expand_dims(normalized, axis=0), axis=-1)
        
        return preprocessed
    
    @staticmethod
    def detect_faces(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in image using OpenCV"""
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Convert to grayscale for detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        return [(x, y, w, h) for (x, y, w, h) in faces]
    
    @staticmethod
    def crop_face(image: np.ndarray, face_coords: Tuple[int, int, int, int]) -> np.ndarray:
        """Crop face from image"""
        x, y, w, h = face_coords
        return image[y:y+h, x:x+w]
    
    @staticmethod
    def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
        """Convert PIL image to OpenCV format"""
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    @staticmethod
    def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
        """Convert OpenCV image to PIL format"""
        return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

class TextUtils:
    """Utility functions for text processing"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and preprocess text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep emoticons
        text = re.sub(r'[^\w\s:)(]', '', text)
        
        return text
    
    @staticmethod
    def tokenize_text(text: str, tokenizer_name: str = "distilbert-base-uncased", max_length: int = 512) -> Dict:
        """Tokenize text for BERT-based models"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            
            # Clean text first
            clean_text = TextUtils.clean_text(text)
            
            # Tokenize
            encoded = tokenizer(
                clean_text,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return encoded
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            return None
    
    @staticmethod
    def extract_emotions_from_text(text: str) -> Dict[str, float]:
        """Extract emotion-related keywords from text"""
        emotion_keywords = {
            'happy': ['happy', 'joy', 'glad', 'excited', 'cheerful', 'delighted', 'pleased'],
            'sad': ['sad', 'depressed', 'unhappy', 'miserable', 'gloomy', 'down'],
            'angry': ['angry', 'mad', 'furious', 'rage', 'irritated', 'annoyed'],
            'fear': ['scared', 'afraid', 'terrified', 'anxious', 'worried', 'nervous'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'sick', 'nauseated'],
            'neutral': ['okay', 'fine', 'normal', 'usual', 'regular']
        }
        
        text_lower = text.lower()
        emotion_scores = {}
        
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            emotion_scores[emotion] = score
        
        # Normalize scores
        total = sum(emotion_scores.values())
        if total > 0:
            emotion_scores = {k: v/total for k, v in emotion_scores.items()}
        else:
            emotion_scores = {k: 1/len(emotion_scores) for k in emotion_scores}
        
        return emotion_scores

class ModelUtils:
    """Utility functions for model operations"""
    
    @staticmethod
    def save_model_state(model, optimizer, epoch: int, loss: float, path: str):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, path)
        logger.info(f"Model checkpoint saved to {path}")
    
    @staticmethod
    def load_model_state(model, optimizer, path: str) -> Tuple[int, float]:
        """Load model checkpoint"""
        if os.path.exists(path):
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            logger.info(f"Model checkpoint loaded from {path}")
            return epoch, loss
        else:
            logger.warning(f"No checkpoint found at {path}")
            return 0, float('inf')
    
    @staticmethod
    def get_device() -> torch.device:
        """Get available device (CUDA if available, else CPU)"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU device")
        return device

class DataUtils:
    """Utility functions for data operations"""
    
    @staticmethod
    def create_food_mood_mapping() -> pd.DataFrame:
        """Create sample food-mood mapping data"""
        data = {
            'food_item': [
                'Chocolate', 'Ice Cream', 'Pizza', 'Salad', 'Soup', 'Coffee', 'Tea', 
                'Banana', 'Nuts', 'Yogurt', 'Berries', 'Avocado', 'Salmon', 'Spinach',
                'Dark Chocolate', 'Green Tea', 'Oatmeal', 'Turkey', 'Sweet Potato'
            ],
            'happy': [0.9, 0.8, 0.7, 0.5, 0.3, 0.6, 0.4, 0.7, 0.5, 0.6, 0.8, 0.6, 0.7, 0.5, 0.8, 0.5, 0.6, 0.5, 0.6],
            'sad': [0.8, 0.9, 0.6, 0.3, 0.8, 0.7, 0.8, 0.6, 0.4, 0.5, 0.7, 0.4, 0.6, 0.5, 0.7, 0.7, 0.7, 0.5, 0.5],
            'angry': [0.3, 0.4, 0.5, 0.7, 0.4, 0.3, 0.8, 0.4, 0.6, 0.3, 0.5, 0.5, 0.4, 0.6, 0.4, 0.8, 0.5, 0.4, 0.4],
            'fear': [0.5, 0.6, 0.3, 0.4, 0.7, 0.4, 0.9, 0.5, 0.3, 0.4, 0.4, 0.3, 0.4, 0.4, 0.3, 0.9, 0.6, 0.4, 0.4],
            'surprise': [0.7, 0.7, 0.5, 0.4, 0.3, 0.5, 0.3, 0.5, 0.4, 0.4, 0.6, 0.5, 0.5, 0.4, 0.6, 0.4, 0.4, 0.4, 0.5],
            'disgust': [0.2, 0.3, 0.2, 0.8, 0.4, 0.2, 0.3, 0.3, 0.2, 0.3, 0.4, 0.4, 0.5, 0.7, 0.3, 0.3, 0.4, 0.3, 0.4],
            'neutral': [0.5, 0.5, 0.5, 0.6, 0.5, 0.5, 0.5, 0.6, 0.5, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6]
        }
        
        df = pd.DataFrame(data)
        return df
    
    @staticmethod
    def save_data_to_csv(data: Union[pd.DataFrame, Dict], filepath: str):
        """Save data to CSV file"""
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")
    
    @staticmethod
    def load_data_from_csv(filepath: str) -> Optional[pd.DataFrame]:
        """Load data from CSV file"""
        try:
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                logger.info(f"Data loaded from {filepath}")
                return df
            else:
                logger.warning(f"File not found: {filepath}")
                return None
        except Exception as e:
            logger.error(f"Error loading data from {filepath}: {e}")
            return None

class ConfigUtils:
    """Utility functions for configuration management"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            return {}
    
    @staticmethod
    def save_config(config: Dict, config_path: str):
        """Save configuration to JSON file"""
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Error saving config to {config_path}: {e}")
    
    @staticmethod
    def get_default_config() -> Dict:
        """Get default configuration"""
        return {
            "model": {
                "text_model_name": "distilbert-base-uncased",
                "image_model_name": "mobilenetv2",
                "max_text_length": 512,
                "image_size": [48, 48],
                "num_emotions": 7
            },
            "training": {
                "batch_size": 32,
                "learning_rate": 2e-5,
                "num_epochs": 10,
                "validation_split": 0.2
            },
            "recommendation": {
                "top_k": 10,
                "similarity_threshold": 0.7
            },
            "paths": {
                "data_dir": "data/",
                "models_dir": "models/",
                "logs_dir": "logs/"
            }
        }