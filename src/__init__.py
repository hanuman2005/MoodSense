"""
MoodSense: Multi-Modal Emotion Recognition & Recommendation System

A comprehensive AI system for emotion recognition and personalized recommendations.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main modules for easier access
from .data_loader import DataLoader
from .model_trainer import ModelTrainer
from .recommender import MusicRecommender, FoodRecommender
from .utils import EmotionUtils, ImageUtils, TextUtils

__all__ = [
    "DataLoader",
    "ModelTrainer", 
    "MusicRecommender",
    "FoodRecommender",
    "EmotionUtils",
    "ImageUtils", 
    "TextUtils"
]