import asyncio
import time
import logging
import requests
import numpy as np
import psycopg2
from datetime import datetime
from transformers import pipeline
import cv2
import pytesseract
import speech_recognition as sr
import feedparser

# -------------------------
# Logging Configuration
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -------------------------
# Database Configuration
# -------------------------
DB_CONFIG = {
    "dbname": "sentiment_db",
    "user": "your_user",
    "password": "your_password",
    "host": "localhost",
    "port": "5432",
}

# -------------------------
# Global Configurations
# -------------------------
UPDATE_TEXT_INTERVAL = 600     # Text updates every 10 min
UPDATE_IMAGE_INTERVAL = 43200  # Images (memes) update twice a day
UPDATE_AUDIO_INTERVAL = 86400  # Podcasts update once per day
PODCAST_RSS_FEED = "https://example.com/podcast/feed"  # Replace with actual feed

NLP_MODEL = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -------------------------
# Database Connection
# -------------------------
def store_sentiment(text_sent, image_sent, audio_sent, final_sent):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        query = """
            INSERT INTO sentiment_scores (text_sentiment, image_sentiment, audio_sentiment, final_sentiment)
            VALUES (%s, %s, %s, %s);
        """
        cursor.execute(query, (text_sent, image_sent, audio_sent, final_sent))
        conn.commit()
        cursor.close()
        conn.close()
        logging.info("Sentiment scores saved to database.")
    except Exception as e:
        logging.error(f"Database error: {e}")

# -------------------------
# Sentiment Processing Classes
# -------------------------
class AspectBasedSentiment:
    def __init__(self):
        self.model = NLP_MODEL

    def analyze_text(self, text):
        if not text.strip():
            return 0.0  # Default to neutral
        result = self.model(text)[0]
        return self._normalize_score(result["label"])

    @staticmethod
    def _normalize_score(label):
        mapping = {"1 star": -1.0, "2 stars": -0.5, "3 stars": 0.0, "4 stars": 0.5, "5 stars": 1.0}
        return mapping.get(label, 0.0)

class MultiModalSentimentExtractor:
    def __init__(self):
        self.text_analyzer = AspectBasedSentiment()
        self.last_text_update = None
        self.last_image_update = None
        self.last_audio_update = None
        self.last_podcast_title = None

    def extract_text_sentiment(self, text_sources):
        scores = [self.text_analyzer.analyze_text(text) for text in text_sources]
        return np.mean(scores) if scores else 0.0

    def extract_audio_sentiment(self, audio_path):
        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(audio_path) as source:
                audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            return self.text_analyzer.analyze_text(text)
        except Exception as e:
            logging.error(f"Audio sentiment extraction failed: {e}")
            return 0.0

    def extract_image_sentiment(self, image_path):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
        try:
            img = cv2.imread(image_path)
            text = pytesseract.image_to_string(img)
            return self.text_analyzer.analyze_text(text)
        except Exception as e:
            logging.error(f"Image sentiment extraction failed: {e}")
            return 0.0

    def check_new_podcast(self):
        try:
            feed = feedparser.parse(PODCAST_RSS_FEED)
            latest_entry = feed.entries[0] if feed.entries else None
            if latest_entry and latest_entry.title != self.last_podcast_title:
                self.last_podcast_title = latest_entry.title
                return latest_entry.links[0]  # Return podcast URL if new episode found
        except Exception as e:
            logging.error(f"Error checking podcast feed: {e}")
        return None

    async def update_sentiment(self):
        while True:
            current_time = time.time()
            
            text_sentiment = 0.0
            image_sentiment = 0.0
            audio_sentiment = 0.0

            # Update Text Sentiment (every 10 min)
            if not self.last_text_update or (current_time - self.last_text_update) > UPDATE_TEXT_INTERVAL:
                logging.info("Updating text sentiment...")
                text_sources = [
                    "Bitcoin hits all-time high!", 
                    "Regulators discuss crypto ban.", 
                    "Ethereum 2.0 upgrade expected next quarter."
                ]
                text_sentiment = self.extract_text_sentiment(text_sources)
                self.last_text_update = current_time

            # Update Image Sentiment (every 12 hours)
            if not self.last_image_update or (current_time - self.last_image_update) > UPDATE_IMAGE_INTERVAL:
                logging.info("Updating image sentiment...")
                image_path = "meme_sample.jpg"
                image_sentiment = self.extract_image_sentiment(image_path)
                self.last_image_update = current_time

            # Update Audio Sentiment (every 24 hours OR if new podcast found)
            new_podcast_url = self.check_new_podcast()
            if new_podcast_url or (not self.last_audio_update or (current_time - self.last_audio_update) > UPDATE_AUDIO_INTERVAL):
                logging.info(f"Updating audio sentiment... New podcast detected: {bool(new_podcast_url)}")
                audio_path = "podcast_sample.wav"
                audio_sentiment = self.extract_audio_sentiment(audio_path)
                self.last_audio_update = current_time

            # Compute Final Sentiment Score
            final_sentiment_score = np.mean([text_sentiment, image_sentiment, audio_sentiment])

            # Store in Database
            store_sentiment(text_sentiment, image_sentiment, audio_sentiment, final_sentiment_score)

            logging.info(f"Final Sentiment Score: {final_sentiment_score:.3f}")
            await asyncio.sleep(60)  # Check every minute

# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    extractor = MultiModalSentimentExtractor()
    asyncio.run(extractor.update_sentiment())
