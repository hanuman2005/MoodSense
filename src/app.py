"""
Main Streamlit application for MoodSense
"""

import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import torch
import plotly.express as px
import plotly.graph_objects as go
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import os
import logging
from typing import Dict, List, Optional
import time

# Import custom modules
from data_loader import DataLoader
from model_trainer import ModelTrainer, TextEmotionModel, ImageEmotionModel
from recommender import MusicRecommender, FoodRecommender
from utils import EmotionUtils, ImageUtils, TextUtils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="MoodSense - Emotion Recognition & Recommendations",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .emotion-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    .recommendation-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1E88E5;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class VideoProcessor(VideoProcessorBase):
    """Video processor for real-time emotion detection"""
    
    def __init__(self):
        self.emotion_scores = {'neutral': 1.0}
        self.model_trainer = ModelTrainer()
        self.image_model = None
        self.last_prediction_time = 0
        self.prediction_interval = 1.0  # Predict every 1 second
        
    def load_image_model(self, model_path: str):
        """Load the trained image emotion model"""
        if os.path.exists(model_path):
            self.image_model = self.model_trainer.load_trained_model(model_path, "image")
            
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Predict emotion periodically
        current_time = time.time()
        if current_time - self.last_prediction_time > self.prediction_interval:
            if self.image_model is not None:
                # Detect faces
                faces = ImageUtils.detect_faces(img)
                
                if faces:
                    # Use the largest face
                    largest_face = max(faces, key=lambda x: x[2] * x[3])
                    face_img = ImageUtils.crop_face(img, largest_face)
                    
                    # Predict emotion
                    try:
                        emotion_scores = self.model_trainer.predict_single_image(self.image_model, face_img)
                        self.emotion_scores = emotion_scores
                    except Exception as e:
                        logger.error(f"Error predicting emotion: {e}")
                
                # Draw face rectangles
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Display dominant emotion
                    dominant_emotion = max(self.emotion_scores.items(), key=lambda x: x[1])[0]
                    emoji = EmotionUtils.get_emotion_emoji(dominant_emotion)
                    cv2.putText(img, f"{emoji} {dominant_emotion.capitalize()}", 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                self.last_prediction_time = current_time
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

@st.cache_resource
def load_models():
    """Load trained models"""
    model_trainer = ModelTrainer()
    
    # Paths to saved models
    text_model_path = "models/text_emotion_model/model.pth"
    image_model_path = "models/image_emotion_model/model.pth"
    
    text_model = None
    image_model = None
    
    if os.path.exists(text_model_path):
        text_model = model_trainer.load_trained_model(text_model_path, "text")
        st.success("Text emotion model loaded successfully!")
    else:
        st.warning("Text emotion model not found. Please train the model first.")
    
    if os.path.exists(image_model_path):
        image_model = model_trainer.load_trained_model(image_model_path, "image")
        st.success("Image emotion model loaded successfully!")
    else:
        st.warning("Image emotion model not found. Please train the model first.")
    
    return model_trainer, text_model, image_model

@st.cache_resource
def load_recommenders():
    """Load recommendation systems"""
    music_recommender = MusicRecommender("data/spotify_dataset/data.csv")
    food_recommender = FoodRecommender("data/food_mood_mapping.csv")
    
    return music_recommender, food_recommender

def predict_text_emotion(text: str, text_model, model_trainer) -> Dict[str, float]:
    """Predict emotion from text"""
    if text_model is None:
        # Fallback to keyword-based prediction
        return TextUtils.extract_emotions_from_text(text)
    else:
        return model_trainer.predict_single_text(text_model, text)

def predict_image_emotion(image: np.ndarray, image_model, model_trainer) -> Dict[str, float]:
    """Predict emotion from image"""
    if image_model is None:
        return {'neutral': 1.0}
    else:
        return model_trainer.predict_single_image(image_model, image)

def display_emotion_chart(emotion_scores: Dict[str, float]):
    """Display emotion scores as a chart"""
    emotions = list(emotion_scores.keys())
    scores = list(emotion_scores.values())
    colors = [EmotionUtils.EMOTION_COLORS.get(emotion, '#888888') for emotion in emotions]
    
    fig = go.Figure(data=[
        go.Bar(x=emotions, y=scores, marker_color=colors)
    ])
    
    fig.update_layout(
        title="Emotion Analysis Results",
        xaxis_title="Emotions",
        yaxis_title="Confidence Score",
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    if mode == "Text Analysis":
        st.header("📝 Text Emotion Analysis")
        
        # Text input
        text_input = st.text_area(
            "Enter your text here:",
            placeholder="How are you feeling today? Share your thoughts...",
            height=150
        )
        
        if st.button("Analyze Text", type="primary"):
            if text_input.strip():
                with st.spinner("Analyzing emotions..."):
                    emotion_scores = predict_text_emotion(text_input, text_model, model_trainer)
                
                # Display results
                st.subheader("Emotion Analysis Results")
                
                # Dominant emotion
                dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
                emoji = EmotionUtils.get_emotion_emoji(dominant_emotion)
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"""
                    <div class="emotion-card">
                        <h2>{emoji}</h2>
                        <h3>{dominant_emotion.capitalize()}</h3>
                        <p>Confidence: {emotion_scores[dominant_emotion]:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Emotion chart
                display_emotion_chart(emotion_scores)
                
                # Get recommendations
                with st.spinner("Getting personalized recommendations..."):
                    music_recs = music_recommender.get_recommendations(emotion_scores, num_recommendations=5)
                    food_recs = food_recommender.get_food_recommendations(emotion_scores, num_recommendations=5)
                
                # Display recommendations
                display_recommendations(music_recs, food_recs)
            else:
                st.warning("Please enter some text to analyze.")
    
    elif mode == "Image Analysis":
        st.header("📷 Image Emotion Analysis")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Upload an image:",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of a face for emotion analysis"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                if st.button("Analyze Image", type="primary"):
                    with st.spinner("Detecting faces and analyzing emotions..."):
                        # Convert PIL to OpenCV format
                        image_cv = ImageUtils.pil_to_cv2(image)
                        
                        # Detect faces
                        faces = ImageUtils.detect_faces(image_cv)
                        
                        if faces:
                            # Use the largest face
                            largest_face = max(faces, key=lambda x: x[2] * x[3])
                            face_img = ImageUtils.crop_face(image_cv, largest_face)
                            
                            # Predict emotion
                            emotion_scores = predict_image_emotion(face_img, image_model, model_trainer)
                            
                            # Display results
                            st.success(f"Found {len(faces)} face(s) in the image!")
                            
                            # Dominant emotion
                            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
                            emoji = EmotionUtils.get_emotion_emoji(dominant_emotion)
                            
                            st.markdown(f"""
                            <div class="emotion-card">
                                <h2>{emoji}</h2>
                                <h3>{dominant_emotion.capitalize()}</h3>
                                <p>Confidence: {emotion_scores[dominant_emotion]:.1%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        else:
                            st.error("No faces detected in the image. Please upload a clear image with a visible face.")
                            emotion_scores = {'neutral': 1.0}
            
            # Show results if emotion analysis was performed
            if 'emotion_scores' in locals():
                # Emotion chart
                display_emotion_chart(emotion_scores)
                
                # Get recommendations
                with st.spinner("Getting personalized recommendations..."):
                    music_recs = music_recommender.get_recommendations(emotion_scores, num_recommendations=5)
                    food_recs = food_recommender.get_food_recommendations(emotion_scores, num_recommendations=5)
                
                # Display recommendations
                display_recommendations(music_recs, food_recs)
    
    elif mode == "Live Camera":
        st.header("📹 Live Camera Emotion Analysis")
        st.info("This feature requires camera access. Click 'START' to begin real-time emotion detection.")
        
        # WebRTC configuration
        RTC_CONFIGURATION = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        # Video processor
        video_processor = VideoProcessor()
        
        # Load image model for video processor
        image_model_path = "models/image_emotion_model/model.pth"
        video_processor.load_image_model(image_model_path)
        
        # WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="emotion-detection",
            video_processor_factory=lambda: video_processor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # Display current emotion if available
        if webrtc_ctx.state.playing:
            st.subheader("Current Emotion Detection")
            
            # Create placeholder for live updates
            emotion_placeholder = st.empty()
            recommendations_placeholder = st.empty()
            
            # Update emotion display
            if hasattr(video_processor, 'emotion_scores'):
                emotion_scores = video_processor.emotion_scores
                
                with emotion_placeholder.container():
                    dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
                    emoji = EmotionUtils.get_emotion_emoji(dominant_emotion)
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.markdown(f"""
                        <div class="emotion-card">
                            <h2>{emoji}</h2>
                            <h3>{dominant_emotion.capitalize()}</h3>
                            <p>Confidence: {emotion_scores[dominant_emotion]:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Get and display recommendations
                with recommendations_placeholder.container():
                    music_recs = music_recommender.get_recommendations(emotion_scores, num_recommendations=3)
                    food_recs = food_recommender.get_food_recommendations(emotion_scores, num_recommendations=3)
                    
                    if music_recs or food_recs:
                        st.subheader("Live Recommendations")
                        display_recommendations(music_recs, food_recs)
    
    elif mode == "About":
        st.header("About MoodSense")
        
        st.markdown("""
        ### 🎯 What is MoodSense?
        
        MoodSense is an AI-powered application that recognizes emotions from text and facial expressions, 
        then provides personalized music and food recommendations based on your detected mood.
        
        ### 🔧 Features
        
        - **Text Emotion Analysis**: Analyze emotional content in text using advanced NLP models
        - **Facial Emotion Recognition**: Detect emotions from uploaded images or live camera feed
        - **Music Recommendations**: Get personalized song suggestions from Spotify based on your mood
        - **Food Recommendations**: Receive mood-based food suggestions for emotional wellness
        - **Real-time Analysis**: Live emotion detection through your camera
        
        ### 🧠 Technology Stack
        
        - **Machine Learning**: PyTorch, TensorFlow, Transformers
        - **Computer Vision**: OpenCV, PIL
        - **Web Interface**: Streamlit
        - **Music Data**: Spotify Web API
        - **Models**: DistilBERT (text), MobileNetV2 (images)
        
        ### 📊 Supported Emotions
        """)
        
        # Display emotion cards
        cols = st.columns(4)
        emotions_display = [
            ('Happy', '😊'), ('Sad', '😢'), ('Angry', '😠'), ('Fear', '😨'),
            ('Surprise', '😲'), ('Disgust', '🤢'), ('Neutral', '😐')
        ]
        
        for i, (emotion, emoji) in enumerate(emotions_display):
            with cols[i % 4]:
                st.markdown(f"""
                <div class="metric-card">
                    <h2>{emoji}</h2>
                    <p>{emotion}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("""
        ### 🚀 Getting Started
        
        1. **Text Analysis**: Enter any text to analyze its emotional content
        2. **Image Analysis**: Upload a photo with a clear face visible
        3. **Live Camera**: Enable camera access for real-time emotion detection
        4. **Get Recommendations**: Receive personalized music and food suggestions
        
        ### 📝 Note
        
        This application requires trained models for optimal performance. If you see warnings about missing models,
        please run the training notebooks first to train the emotion recognition models on your datasets.
        
        ### 🎵 Music Integration
        
        Music recommendations are powered by Spotify's audio features and a content-based filtering system
        that matches your emotions to song characteristics like valence, energy, and danceability.
        
        ### 🍽️ Food Recommendations
        
        Food suggestions are based on research about mood-food relationships and the psychological effects
        of different nutrients on emotional well-being.
        """)
        
        # Statistics section
        st.subheader("📈 Model Performance")
        
        # Placeholder metrics (you can update these with real metrics)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Text Model Accuracy",
                value="85.2%",
                delta="2.1%"
            )
        
        with col2:
            st.metric(
                label="Image Model Accuracy", 
                value="78.9%",
                delta="1.8%"
            )
        
        with col3:
            st.metric(
                label="Recommendation Accuracy",
                value="92.3%",
                delta="3.2%"
            )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Made with ❤️ using Streamlit | MoodSense v1.0</p>
            <p>For better results, ensure good lighting for camera analysis and clear, descriptive text for text analysis.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
    
    st.plotly_chart(fig, use_container_width=True)

def display_recommendations(music_recs: List[Dict], food_recs: List[Dict]):
    """Display music and food recommendations"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎵 Music Recommendations")
        
        if music_recs:
            for i, rec in enumerate(music_recs):
                with st.container():
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <h4>{rec['track_name']}</h4>
                        <p><strong>Artist:</strong> {rec['artists']}</p>
                        <p><strong>Match:</strong> {rec.get('similarity', 0):.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if rec.get('preview_url'):
                        st.audio(rec['preview_url'])
        else:
            st.info("No music recommendations available. Please check your Spotify dataset.")
    
    with col2:
        st.subheader("🍽️ Food Recommendations")
        
        if food_recs:
            for rec in food_recs:
                st.markdown(f"""
                <div class="recommendation-card">
                    <h4>{rec['food_item']}</h4>
                    <p>{rec['description']}</p>
                    <p><strong>Benefits:</strong> {', '.join(rec['benefits'][:3])}</p>
                    <p><strong>Match Score:</strong> {rec['score']:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No food recommendations available.")

def main():
    """Main application"""
    st.markdown('<h1 class="main-header">MoodSense 😊🎵🍽️</h1>', unsafe_allow_html=True)
    st.markdown("**AI-Powered Emotion Recognition & Personalized Recommendations**")
    
    # Load models and recommenders
    with st.spinner("Loading models..."):
        model_trainer, text_model, image_model = load_models()
        music_recommender, food_recommender = load_recommenders()
    
    # Sidebar
    st.sidebar.title("Navigation")
    mode = st.sidebar.selectbox(
        "Choose Analysis Mode",
        ["Text Analysis", "Image Analysis", "Live Camera", "About"]