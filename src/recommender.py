"""
Recommendation system for music and food based on emotions
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
from typing import List, Dict, Tuple, Optional
import logging
from dotenv import load_dotenv
import random

from .utils import EmotionUtils

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class MusicRecommender:
    """Music recommendation system based on emotions and Spotify data"""
    
    def __init__(self, spotify_data_path: str = None):
        self.spotify_data = None
        self.scaler = StandardScaler()
        self.audio_features = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                              'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
        
        # Spotify API setup
        self.spotify_client = self._setup_spotify_client()
        
        # Load Spotify data if provided
        if spotify_data_path and os.path.exists(spotify_data_path):
            self.load_spotify_data(spotify_data_path)
        
        # Emotion to audio features mapping
        self.emotion_to_features = self._create_emotion_feature_mapping()
    
    def _setup_spotify_client(self) -> Optional[spotipy.Spotify]:
        """Setup Spotify API client"""
        try:
            client_id = os.getenv('SPOTIFY_CLIENT_ID')
            client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
            
            if client_id and client_secret:
                client_credentials_manager = SpotifyClientCredentials(
                    client_id=client_id, client_secret=client_secret
                )
                spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
                logger.info("Spotify API client initialized successfully")
                return spotify
            else:
                logger.warning("Spotify API credentials not found in environment variables")
                return None
        except Exception as e:
            logger.error(f"Error setting up Spotify client: {e}")
            return None
    
    def _create_emotion_feature_mapping(self) -> Dict[str, Dict[str, float]]:
        """Create mapping from emotions to audio features"""
        return {
            'happy': {
                'acousticness': 0.2, 'danceability': 0.8, 'energy': 0.8,
                'instrumentalness': 0.1, 'liveness': 0.3, 'loudness': -5,
                'speechiness': 0.1, 'tempo': 120, 'valence': 0.9
            },
            'sad': {
                'acousticness': 0.7, 'danceability': 0.3, 'energy': 0.2,
                'instrumentalness': 0.6, 'liveness': 0.1, 'loudness': -15,
                'speechiness': 0.05, 'tempo': 80, 'valence': 0.1
            },
            'angry': {
                'acousticness': 0.1, 'danceability': 0.6, 'energy': 0.9,
                'instrumentalness': 0.3, 'liveness': 0.4, 'loudness': -3,
                'speechiness': 0.3, 'tempo': 140, 'valence': 0.2
            },
            'fear': {
                'acousticness': 0.5, 'danceability': 0.2, 'energy': 0.3,
                'instrumentalness': 0.8, 'liveness': 0.1, 'loudness': -12,
                'speechiness': 0.05, 'tempo': 90, 'valence': 0.2
            },
            'surprise': {
                'acousticness': 0.3, 'danceability': 0.7, 'energy': 0.7,
                'instrumentalness': 0.2, 'liveness': 0.5, 'loudness': -6,
                'speechiness': 0.2, 'tempo': 110, 'valence': 0.7
            },
            'disgust': {
                'acousticness': 0.4, 'danceability': 0.2, 'energy': 0.4,
                'instrumentalness': 0.5, 'liveness': 0.1, 'loudness': -10,
                'speechiness': 0.1, 'tempo': 85, 'valence': 0.1
            },
            'neutral': {
                'acousticness': 0.5, 'danceability': 0.5, 'energy': 0.5,
                'instrumentalness': 0.3, 'liveness': 0.2, 'loudness': -8,
                'speechiness': 0.1, 'tempo': 100, 'valence': 0.5
            }
        }
    
    def load_spotify_data(self, data_path: str):
        """Load Spotify dataset"""
        try:
            self.spotify_data = pd.read_csv(data_path)
            
            # Clean column names
            self.spotify_data.columns = self.spotify_data.columns.str.lower().str.replace(' ', '_')
            
            # Ensure required columns exist
            required_cols = ['track_name', 'artists'] + self.audio_features
            missing_cols = [col for col in required_cols if col not in self.spotify_data.columns]
            
            if missing_cols:
                logger.warning(f"Missing columns in Spotify data: {missing_cols}")
            
            # Scale audio features
            feature_data = self.spotify_data[self.audio_features].fillna(0)
            self.spotify_data[self.audio_features] = self.scaler.fit_transform(feature_data)
            
            logger.info(f"Loaded Spotify data with {len(self.spotify_data)} tracks")
            
        except Exception as e:
            logger.error(f"Error loading Spotify data: {e}")
            self.spotify_data = None
    
    def get_emotion_based_features(self, emotion_scores: Dict[str, float]) -> Dict[str, float]:
        """Get target audio features based on emotion scores"""
        target_features = {feature: 0.0 for feature in self.audio_features}
        
        for emotion, score in emotion_scores.items():
            if emotion in self.emotion_to_features:
                emotion_features = self.emotion_to_features[emotion]
                for feature in self.audio_features:
                    if feature in emotion_features:
                        target_features[feature] += score * emotion_features[feature]
        
        return target_features
    
    def recommend_songs_from_dataset(self, emotion_scores: Dict[str, float], 
                                   num_recommendations: int = 10) -> List[Dict]:
        """Recommend songs from loaded dataset based on emotions"""
        if self.spotify_data is None or self.spotify_data.empty:
            logger.error("No Spotify data loaded")
            return []
        
        try:
            # Get target features based on emotions
            target_features = self.get_emotion_based_features(emotion_scores)
            target_vector = np.array([target_features[f] for f in self.audio_features]).reshape(1, -1)
            target_vector = self.scaler.transform(target_vector)
            
            # Calculate similarities
            feature_matrix = self.spotify_data[self.audio_features].values
            similarities = cosine_similarity(target_vector, feature_matrix)[0]
            
            # Get top recommendations
            top_indices = np.argsort(similarities)[-num_recommendations:][::-1]
            
            recommendations = []
            for idx in top_indices:
                song_data = self.spotify_data.iloc[idx]
                recommendations.append({
                    'track_name': song_data.get('track_name', 'Unknown'),
                    'artists': song_data.get('artists', 'Unknown'),
                    'similarity': float(similarities[idx]),
                    'spotify_id': song_data.get('track_id', ''),
                    'preview_url': song_data.get('preview_url', ''),
                    'external_url': song_data.get('external_urls', ''),
                    'audio_features': {f: song_data.get(f, 0) for f in self.audio_features}
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error recommending songs from dataset: {e}")
            return []
    
    def recommend_songs_from_spotify_api(self, emotion_scores: Dict[str, float],
                                        num_recommendations: int = 10) -> List[Dict]:
        """Recommend songs using Spotify API based on emotions"""
        if not self.spotify_client:
            logger.error("Spotify API client not available")
            return []
        
        try:
            # Get dominant emotion
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
            
            # Get target features
            target_features = self.get_emotion_based_features(emotion_scores)
            
            # Create search query based on emotion
            emotion_genres = {
                'happy': ['pop', 'dance', 'funk', 'disco'],
                'sad': ['acoustic', 'folk', 'indie', 'blues'],
                'angry': ['rock', 'metal', 'punk', 'alternative'],
                'fear': ['ambient', 'classical', 'instrumental'],
                'surprise': ['electronic', 'experimental', 'jazz'],
                'disgust': ['grunge', 'industrial', 'alternative'],
                'neutral': ['indie', 'alternative', 'pop']
            }
            
            genres = emotion_genres.get(dominant_emotion, ['pop'])
            selected_genre = random.choice(genres)
            
            # Get recommendations from Spotify
            recommendations_data = self.spotify_client.recommendations(
                seed_genres=[selected_genre],
                limit=num_recommendations,
                target_acousticness=target_features.get('acousticness', 0.5),
                target_danceability=target_features.get('danceability', 0.5),
                target_energy=target_features.get('energy', 0.5),
                target_valence=target_features.get('valence', 0.5),
                target_tempo=target_features.get('tempo', 100)
            )
            
            recommendations = []
            for track in recommendations_data['tracks']:
                # Get audio features for each track
                audio_features = self.spotify_client.audio_features(track['id'])
                if audio_features and audio_features[0]:
                    features = audio_features[0]
                    recommendations.append({
                        'track_name': track['name'],
                        'artists': ', '.join([artist['name'] for artist in track['artists']]),
                        'spotify_id': track['id'],
                        'preview_url': track['preview_url'],
                        'external_url': track['external_urls']['spotify'],
                        'audio_features': {
                            'acousticness': features['acousticness'],
                            'danceability': features['danceability'],
                            'energy': features['energy'],
                            'valence': features['valence'],
                            'tempo': features['tempo']
                        },
                        'similarity': self._calculate_similarity(target_features, features)
                    })
            
            return sorted(recommendations, key=lambda x: x['similarity'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error recommending songs from Spotify API: {e}")
            return []
    
    def _calculate_similarity(self, target_features: Dict, spotify_features: Dict) -> float:
        """Calculate similarity between target and spotify features"""
        try:
            target_vals = [target_features.get(f, 0) for f in ['acousticness', 'danceability', 'energy', 'valence']]
            spotify_vals = [spotify_features.get(f, 0) for f in ['acousticness', 'danceability', 'energy', 'valence']]
            
            similarity = cosine_similarity([target_vals], [spotify_vals])[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def get_recommendations(self, emotion_scores: Dict[str, float], 
                          num_recommendations: int = 10, use_api: bool = False) -> List[Dict]:
        """Get music recommendations based on emotions"""
        if use_api and self.spotify_client:
            return self.recommend_songs_from_spotify_api(emotion_scores, num_recommendations)
        else:
            return self.recommend_songs_from_dataset(emotion_scores, num_recommendations)

class FoodRecommender:
    """Food recommendation system based on emotions"""
    
    def __init__(self, food_data_path: str = None):
        self.food_data = None
        
        # Load food-mood mapping data
        if food_data_path and os.path.exists(food_data_path):
            self.load_food_data(food_data_path)
        else:
            self.create_default_food_data()
    
    def load_food_data(self, data_path: str):
        """Load food-mood mapping data"""
        try:
            self.food_data = pd.read_csv(data_path)
            logger.info(f"Loaded food data with {len(self.food_data)} items")
        except Exception as e:
            logger.error(f"Error loading food data: {e}")
            self.create_default_food_data()
    
    def create_default_food_data(self):
        """Create default food-mood mapping"""
        from .utils import DataUtils
        self.food_data = DataUtils.create_food_mood_mapping()
        logger.info("Created default food-mood mapping data")
    
    def get_food_recommendations(self, emotion_scores: Dict[str, float], 
                               num_recommendations: int = 5) -> List[Dict]:
        """Get food recommendations based on emotions"""
        if self.food_data is None or self.food_data.empty:
            return []
        
        try:
            # Calculate similarity scores for each food item
            emotion_columns = [col for col in self.food_data.columns if col in EmotionUtils.EMOTION_LABELS]
            
            recommendations = []
            for _, food_row in self.food_data.iterrows():
                # Calculate weighted score based on emotion probabilities
                food_score = 0.0
                for emotion, score in emotion_scores.items():
                    if emotion in emotion_columns:
                        food_score += score * food_row[emotion]
                
                recommendations.append({
                    'food_item': food_row['food_item'],
                    'score': food_score,
                    'emotion_match': {emotion: food_row.get(emotion, 0) for emotion in emotion_columns},
                    'description': self._get_food_description(food_row['food_item'], emotion_scores),
                    'benefits': self._get_food_benefits(food_row['food_item'])
                })
            
            # Sort by score and return top recommendations
            recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)
            return recommendations[:num_recommendations]
            
        except Exception as e:
            logger.error(f"Error getting food recommendations: {e}")
            return []
    
    def _get_food_description(self, food_item: str, emotion_scores: Dict[str, float]) -> str:
        """Get description for why this food is recommended"""
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
        
        descriptions = {
            'happy': {
                'Chocolate': "Celebrate your happiness with some indulgent chocolate!",
                'Ice Cream': "Sweet treats to match your sweet mood!",
                'Pizza': "Share the joy with friends over pizza!",
                'Berries': "Fresh, vibrant berries for your bright mood!"
            },
            'sad': {
                'Chocolate': "Dark chocolate can help boost your mood with endorphins.",
                'Ice Cream': "Comfort food that provides temporary emotional relief.",
                'Soup': "Warm, nourishing soup for emotional comfort.",
                'Tea': "Calming herbal tea to soothe your emotions."
            },
            'angry': {
                'Tea': "Calming tea to help you relax and de-stress.",
                'Nuts': "Healthy snacks to channel your energy positively.",
                'Salad': "Fresh, cooling foods to balance intense emotions.",
                'Green Tea': "L-theanine in green tea promotes relaxation."
            },
            'fear': {
                'Tea': "Warm, comforting beverages for emotional grounding.",
                'Oatmeal': "Steady, comforting food for stability.",
                'Banana': "Natural mood stabilizers and easy to digest.",
                'Green Tea': "Calming properties to reduce anxiety."
            }
        }
        
        if dominant_emotion in descriptions and food_item in descriptions[dominant_emotion]:
            return descriptions[dominant_emotion][food_item]
        else:
            return f"{food_item} is a great choice for your current mood!"
    
    def _get_food_benefits(self, food_item: str) -> List[str]:
        """Get health/mood benefits of food item"""
        benefits = {
            'Chocolate': ['Releases endorphins', 'Contains antioxidants', 'Boosts serotonin'],
            'Ice Cream': ['Provides calcium', 'Comfort food', 'Quick energy'],
            'Pizza': ['Social food', 'Carbs for energy', 'Can be nutritious with right toppings'],
            'Salad': ['Rich in vitamins', 'Hydrating', 'Low calorie', 'Fiber for digestion'],
            'Soup': ['Hydrating', 'Warming', 'Easy to digest', 'Nutritious'],
            'Coffee': ['Increases alertness', 'Antioxidants', 'May improve mood'],
            'Tea': ['Calming properties', 'Antioxidants', 'Hydrating', 'Various health benefits'],
            'Banana': ['Potassium for heart health', 'Natural sugars', 'Vitamin B6', 'Easy to digest'],
            'Nuts': ['Healthy fats', 'Protein', 'Vitamin E', 'Omega-3 fatty acids'],
            'Yogurt': ['Probiotics for gut health', 'Protein', 'Calcium', 'May improve mood'],
            'Berries': ['Antioxidants', 'Vitamin C', 'Fiber', 'Natural sweetness'],
            'Avocado': ['Healthy monounsaturated fats', 'Fiber', 'Potassium', 'Vitamin K'],
            'Salmon': ['Omega-3 fatty acids', 'High-quality protein', 'Vitamin D', 'Heart-healthy'],
            'Spinach': ['Iron', 'Folate', 'Vitamin K', 'Antioxidants'],
            'Dark Chocolate': ['Antioxidants', 'May improve brain function', 'Heart-healthy', 'Mood booster'],
            'Green Tea': ['L-theanine for relaxation', 'Antioxidants', 'May boost metabolism'],
            'Oatmeal': ['Fiber for sustained energy', 'Heart-healthy', 'Protein', 'Versatile'],
            'Turkey': ['Lean protein', 'Tryptophan for mood', 'B vitamins', 'Low in saturated fat'],
            'Sweet Potato': ['Complex carbs', 'Beta-carotene', 'Fiber', 'Vitamin A']
        }
        
        return benefits.get(food_item, ['Nutritious choice', 'Good for overall health'])
    
    def get_meal_suggestions(self, emotion_scores: Dict[str, float]) -> Dict[str, List[str]]:
        """Get meal suggestions for different times of day"""
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
        
        meal_suggestions = {
            'happy': {
                'breakfast': ['Pancakes with berries', 'Smoothie bowl', 'Avocado toast'],
                'lunch': ['Colorful salad', 'Sushi rolls', 'Mediterranean wrap'],
                'dinner': ['Grilled salmon', 'Pasta with fresh vegetables', 'Stir-fry'],
                'snacks': ['Fresh fruit', 'Yogurt with nuts', 'Dark chocolate']
            },
            'sad': {
                'breakfast': ['Warm oatmeal', 'Hot chocolate', 'Comfort pancakes'],
                'lunch': ['Chicken soup', 'Mac and cheese', 'Grilled cheese sandwich'],
                'dinner': ['Comfort pasta', 'Hearty stew', 'Roast chicken'],
                'snacks': ['Hot tea', 'Cookies', 'Ice cream']
            },
            'angry': {
                'breakfast': ['Green smoothie', 'Yogurt with granola', 'Herbal tea'],
                'lunch': ['Fresh salad', 'Vegetable soup', 'Quinoa bowl'],
                'dinner': ['Grilled vegetables', 'Lean fish', 'Brown rice'],
                'snacks': ['Nuts', 'Herbal tea', 'Fresh vegetables']
            },
            'neutral': {
                'breakfast': ['Balanced cereal', 'Toast with jam', 'Coffee'],
                'lunch': ['Sandwich', 'Regular salad', 'Soup'],
                'dinner': ['Chicken and rice', 'Pasta', 'Vegetables'],
                'snacks': ['Crackers', 'Fruit', 'Nuts']
            }
        }
        
        return meal_suggestions.get(dominant_emotion, meal_suggestions['neutral'])
    
    def add_food_item(self, food_item: str, emotion_scores: Dict[str, float]):
        """Add new food item to the database"""
        if self.food_data is not None:
            new_row = {'food_item': food_item}
            new_row.update(emotion_scores)
            
            self.food_data = pd.concat([self.food_data, pd.DataFrame([new_row])], ignore_index=True)
            logger.info(f"Added new food item: {food_item}")
        
    def save_food_data(self, save_path: str):
        """Save food data to CSV"""
        if self.food_data is not None:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                self.food_data.to_csv(save_path, index=False)
                logger.info(f"Food data saved to {save_path}")
            except Exception as e:
                logger.error(f"Error saving food data: {e}")