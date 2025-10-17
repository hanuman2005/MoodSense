# MoodSense: Multi-Modal Emotion Recognition & Recommendation System

A comprehensive AI system that recognizes emotions from text and facial expressions, then provides personalized music and food recommendations based on detected mood.

## Features

- **Text Emotion Recognition**: Uses fine-tuned BERT model to analyze emotional content in text
- **Facial Emotion Recognition**: Computer vision model to detect emotions from facial expressions
- **Music Recommendations**: Spotify integration with mood-based song suggestions
- **Food Recommendations**: Mood-based food suggestions for emotional wellness
- **Interactive Web Interface**: Streamlit-based user-friendly interface
- **Real-time Processing**: Live camera integration for real-time emotion detection

## Project Structure

```
MoodSense/
├── data/                          # Data storage
│   ├── emotion_text_dataset/      # Text emotion training data
│   ├── fer2013/                   # Facial emotion recognition dataset
│   ├── spotify_dataset/           # Music feature data
│   └── food_mood_mapping.csv      # Food-mood relationship data
├── models/                        # Trained model storage
│   ├── text_emotion_model/        # BERT-based text emotion model
│   ├── image_emotion_model/       # CNN-based facial emotion model
│   └── music_recommender/         # Music recommendation system
├── notebooks/                     # Jupyter notebooks for development
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_text_model_training.ipynb
│   ├── 03_image_model_training.ipynb
│   └── 04_music_recommender_exploration.ipynb
├── src/                          # Source code
│   ├── __init__.py
│   ├── data_loader.py            # Data loading and preprocessing
│   ├── model_trainer.py          # Model training utilities
│   ├── recommender.py            # Recommendation engine
│   ├── app.py                    # Main Streamlit application
│   └── utils.py                  # Helper functions
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── .gitignore                   # Git ignore rules
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd MoodSense
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Variables
Create a `.env` file in the root directory:
```env
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
```

### 5. Data Setup
- Download the emotion text dataset and place in `data/emotion_text_dataset/`
- Download FER-2013 dataset and extract to `data/fer2013/`
- Download Spotify dataset and place in `data/spotify_dataset/`
- The `food_mood_mapping.csv` is created automatically with sample data

### 6. Model Training
Run the Jupyter notebooks in order:
```bash
jupyter notebook notebooks/01_data_preprocessing.ipynb
jupyter notebook notebooks/02_text_model_training.ipynb
jupyter notebook notebooks/03_image_model_training.ipynb
jupyter notebook notebooks/04_music_recommender_exploration.ipynb
```

### 7. Run the Application
```bash
streamlit run src/app.py
```

## Usage

1. **Text Emotion Analysis**: Enter text to analyze emotional content
2. **Image Emotion Recognition**: Upload an image or use live camera feed
3. **Get Recommendations**: Receive personalized music and food suggestions based on detected emotions

## Supported Emotions

- Happy
- Sad
- Angry
- Fear
- Surprise
- Disgust
- Neutral

## Technical Details

### Models Used
- **Text Emotion**: Fine-tuned DistilBERT
- **Image Emotion**: MobileNetV2 with custom classification head
- **Music Recommendation**: Content-based filtering with cosine similarity

### Technologies
- **Backend**: Python, PyTorch, TensorFlow, Transformers
- **Frontend**: Streamlit
- **APIs**: Spotify Web API
- **Computer Vision**: OpenCV, PIL
- **Data Processing**: Pandas, NumPy, Scikit-learn

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- FER-2013 dataset for facial emotion recognition
- Hugging Face Transformers for text processing
- Spotify Web API for music data
- Streamlit for the web interface# MoodSense
