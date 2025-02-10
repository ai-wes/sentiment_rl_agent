#!/usr/bin/env python3
import os
import logging
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from transformers import pipeline
from dotenv import load_dotenv
from ratelimit import limits, sleep_and_retry
from datetime import datetime

load_dotenv()

# Import Tweepy and PRAW for real data ingestion
import tweepy  # Twitter API client
import praw    # Reddit API client

# -------------------------
# Logging Configuration
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


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
# API Credentials (set these as environment variables or constants)
# -------------------------
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
REDDIT_CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT    = os.getenv("REDDIT_USER_AGENT")

# -------------------------
# Rate Limiting Decorators
# -------------------------
@sleep_and_retry
@limits(calls=180, period=900)  # 900 seconds = 15 minutes
def rate_limited_twitter_search(client, query, max_results, tweet_fields):
    print(f"[{datetime.now()}] Making Twitter API request for query: {query}")
    return client.search_recent_tweets(query=query, max_results=max_results, tweet_fields=tweet_fields)

@sleep_and_retry
@limits(calls=60, period=60)  # 60 requests per minute
def rate_limited_reddit_fetch(subreddit, limit):
    print(f"[{datetime.now()}] Making Reddit API request for subreddit: {subreddit.display_name}")
    return list(subreddit.hot(limit=limit))

# -------------------------
# Data Ingestion Functions
# -------------------------
def get_twitter_texts(query="crypto", max_results=10):
    """
    Fetch recent tweets matching a query using Twitter API v2 with rate limiting.
    """
    print(f"\n=== Starting Twitter Data Collection ===")
    print(f"Query: {query}, Max Results: {max_results}")
    
    if not TWITTER_BEARER_TOKEN or TWITTER_BEARER_TOKEN == "YOUR_TWITTER_BEARER_TOKEN":
        logging.error("Twitter Bearer Token not configured")
        print("Error: Twitter Bearer Token not configured")
        return []
    
    client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
    try:
        print("Making rate-limited Twitter API call...")
        response = rate_limited_twitter_search(client, query, max_results, tweet_fields=['text'])
        
        tweets = []
        if response.data:
            for tweet in response.data:
                tweets.append(tweet.text)
                
        print(f"Successfully fetched {len(tweets)} tweets")
        logging.info(f"Fetched {len(tweets)} tweets from Twitter.")
        return tweets
    
    except tweepy.errors.TooManyRequests:
        print("Twitter rate limit exceeded. Waiting before retry...")
        logging.warning("Twitter rate limit exceeded")
        time.sleep(900)  # Wait 15 minutes
        return []
    except Exception as e:
        print(f"Error fetching tweets: {str(e)}")
        logging.error(f"Error fetching tweets: {str(e)}")
        return []
    finally:
        print("=== Twitter Data Collection Complete ===\n")

def get_reddit_texts(subreddit_name="cryptocurrency", limit=10):
    """
    Fetch hot submissions from a subreddit using PRAW with rate limiting.
    """
    print(f"\n=== Starting Reddit Data Collection ===")
    print(f"Subreddit: {subreddit_name}, Limit: {limit}")
    
    if not REDDIT_CLIENT_ID or REDDIT_CLIENT_ID == "YOUR_REDDIT_CLIENT_ID":
        logging.error("Reddit credentials not configured")
        print("Error: Reddit credentials not configured")
        return []
    
    try:
        print("Initializing Reddit client...")
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )
        
        subreddit = reddit.subreddit(subreddit_name)
        texts = []
        
        print("Making rate-limited Reddit API call...")
        submissions = rate_limited_reddit_fetch(subreddit, limit)
        
        for submission in submissions:
            combined_text = f"{submission.title} {submission.selftext}"
            texts.append(combined_text)
            print(f"Processed submission: {submission.title[:50]}...")
        
        print(f"Successfully fetched {len(texts)} submissions")
        logging.info(f"Fetched {len(texts)} submissions from Reddit.")
        return texts
    
    except Exception as e:
        print(f"Error fetching Reddit posts: {str(e)}")
        logging.error(f"Error fetching Reddit posts: {str(e)}")
        return []
    finally:
        print("=== Reddit Data Collection Complete ===\n")

def simulate_audio_ingestion():
    """
    (Placeholder) Simulate transcription from audio sources such as podcasts.
    Replace this function with real speech-to-text implementation if available.
    """
    transcriptions = [
        "Podcasts indicate that the market is experiencing a sudden shift.",
        "Analysts are discussing the impact of recent technological innovations."
    ]
    return transcriptions



def simulate_image_ingestion():
    """
    (Placeholder) Simulate OCR extraction from images (e.g., memes).
    Replace this with an OCR library call if you wish to process real images.
    """
    images_text = [
        "When Bitcoin hits $100k, the memes go wild!",
        "Crypto market crashing: a meme of despair."
    ]
    return images_text

# -------------------------
# Sentiment Processing
# -------------------------
class AspectBasedSentiment:
    """
    Uses a transformer-based sentiment analysis model to obtain sentiment scores.
    For production use, fine-tune on domain-specific data.
    """
    def __init__(self):
        # Using Hugging Face's sentiment-analysis pipeline as a placeholder
        self.model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
        self.aspects = ["regulation", "technology", "market", "manipulation"]
    
    def analyze_text(self, text):
        """
        Analyze text and return a normalized sentiment score.
        """
        # In a real implementation, you could parse out aspect-specific sentiment.
        # Here we simply run the whole text.
        if not text.strip():
            return 0.0
        result = self.model(text)[0]
        return self._normalize_score(result["label"])
    
    def _normalize_score(self, label):
        """
        Normalize sentiment label (e.g. '1 star' to '5 star') to a score between -5 and 5.
        """
        mapping = {
            "1 star": -5.0,
            "2 stars": -2.5,
            "3 stars": 0.0,
            "4 stars": 2.5,
            "5 stars": 5.0
        }
        return mapping.get(label, 0.0)

def aggregate_sentiment():
    """
    Aggregate sentiment scores from multiple sources (Twitter, Reddit, audio, image).
    """
    print("\n=== Starting Sentiment Aggregation ===")
    abs_analyzer = AspectBasedSentiment()
    
    # Gather texts from different sources
    twitter_texts = get_twitter_texts()
    reddit_texts = get_reddit_texts()
    audio_texts = simulate_audio_ingestion()
    image_texts = simulate_image_ingestion()
    
    all_texts = twitter_texts + reddit_texts + audio_texts + image_texts
    print(f"Total texts collected for analysis: {len(all_texts)}")
    
    scores = []
    for i, text in enumerate(all_texts, 1):
        print(f"Analyzing text {i}/{len(all_texts)}...")
        score = abs_analyzer.analyze_text(text)
        scores.append(score)
        print(f"Text {i} sentiment score: {score:.3f}")
    
    if scores:
        aggregated = np.mean(scores)
        print(f"Final aggregated sentiment score: {aggregated:.3f}")
        logging.info(f"Aggregated sentiment score: {aggregated:.3f}")
        return aggregated
    
    print("No scores to aggregate")
    return 0.0

# -------------------------
# Custom RL Environment: GranularSentimentEnv
# -------------------------
class GranularSentimentEnv(gym.Env):
    """
    Custom environment for a dedicated Sentiment RL Agent.
    It outputs an 8-dimensional vector representing granular sentiment features:
      1. Regulation sentiment score
      2. Technology sentiment score
      3. Market sentiment score
      4. Manipulation sentiment score
      5. Sentiment momentum (change from previous aggregated sentiment)
      6. Sentiment volatility (simulated variance)
      7. Confidence score (simulated)
      8. Extreme event flag (binary indicator)
    """
    metadata = {'render_modes': ['human']}
    
    def __init__(self):
        super(GranularSentimentEnv, self).__init__()
        self.observation_space = spaces.Box(low=-10, high=10, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        self.current_step = 0
        self.max_steps = 100
        self.prev_agg_sentiment = aggregate_sentiment()
    
    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.prev_agg_sentiment = aggregate_sentiment()
        return self._get_granular_sentiment_features(), {}
    
    def step(self, action):
        self.current_step += 1
        target = self._simulate_target_features()
        reward = -np.linalg.norm(action - target)
        done = self.current_step >= self.max_steps
        obs = self._get_granular_sentiment_features()
        return obs, reward, done, False, {}
    
    def render(self, mode='human'):
        logging.info(f"GranularSentimentEnv Step: {self.current_step}")
    
    def _get_granular_sentiment_features(self):
        """
        Create an 8-dimensional sentiment feature vector using aggregated sentiment and simulated aspect details.
        """
        agg_sent = aggregate_sentiment()
        # Simulate aspect-specific sentiment scores (in practice, these would be extracted per aspect)
        regulation = np.random.uniform(-5, 5)
        technology = np.random.uniform(-5, 5)
        market = np.random.uniform(-5, 5)
        manipulation = np.random.uniform(-5, 5)
        # Temporal features: momentum is the change in aggregated sentiment
        momentum = agg_sent - self.prev_agg_sentiment
        volatility = np.random.uniform(0, 2)  # simulated volatility
        confidence = np.random.uniform(0, 1)  # simulated confidence
        shock = np.random.choice([0, 1], p=[0.9, 0.1])
        self.prev_agg_sentiment = agg_sent
        features = np.array([regulation, technology, market, manipulation,
                             momentum, volatility, confidence, shock], dtype=np.float32)
        return features
    
    def _simulate_target_features(self):
        """
        Simulate a target sentiment feature vector.
        """
        return np.random.uniform(-1, 1, size=(8,))

# -------------------------
# Custom Feature Extractor for the Sentiment RL Agent
# -------------------------
class GranularSentimentFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 32):
        super(GranularSentimentFeatureExtractor, self).__init__(observation_space, features_dim)
        input_dim = observation_space.shape[0]  # 8-dimensional input
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.model(observations)

# -------------------------
# Training the Sentiment RL Agent
# -------------------------
def train_sentiment_rl_agent():
    env = GranularSentimentEnv()
    policy_kwargs = dict(
        features_extractor_class=GranularSentimentFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=32)
    )
    agent = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs)
    logging.info("Starting training of the Sentiment RL Agent...")
    agent.learn(total_timesteps=50000)
    agent.save("granular_sentiment_rl_model")
    logging.info("Sentiment RL Agent training complete and model saved.")
    return agent

# -------------------------
# Evaluating the Sentiment RL Agent
# -------------------------
def evaluate_sentiment_rl_agent(agent, episodes=3):
    env = GranularSentimentEnv()
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            env.render()
        logging.info(f"Evaluation Episode {ep+1}: Total Reward: {total_reward:.3f}")

# -------------------------
# Main Execution: Pipeline Integration
# -------------------------
if __name__ == "__main__":
    print("\n=== Starting the Full Sentiment Analysis Pipeline ===")
    print(f"Time: {datetime.now()}")
    print("Checking API credentials...")
    
    # Validate credentials
    if TWITTER_BEARER_TOKEN == "YOUR_TWITTER_BEARER_TOKEN":
        print("Warning: Twitter Bearer Token not configured")
    if REDDIT_CLIENT_ID == "YOUR_REDDIT_CLIENT_ID":
        print("Warning: Reddit credentials not configured")
    
    # Test data collection functions
    print("\nTesting data collection...")
    twitter_texts = get_twitter_texts()
    reddit_texts = get_reddit_texts()
    
    if twitter_texts:
        print(f"\nSample Twitter Text (1 of {len(twitter_texts)}):")
        print(twitter_texts[0][:200] + "...")
    if reddit_texts:
        print(f"\nSample Reddit Text (1 of {len(reddit_texts)}):")
        print(reddit_texts[0][:200] + "...")
    
    print("\nCalculating overall sentiment...")
    overall_sentiment = aggregate_sentiment()
    print(f"Overall Aggregated Sentiment: {overall_sentiment:.3f}")
    
    # Train the Sentiment RL Agent with Granular Outputs
    sentiment_agent = train_sentiment_rl_agent()
    
    # Evaluate the trained Sentiment RL Agent
    logging.info("=== Starting Evaluation of the Sentiment RL Agent ===")
    evaluate_sentiment_rl_agent(sentiment_agent, episodes=3)
    
    print("\n=== Full Sentiment Analysis Pipeline Complete ===")
