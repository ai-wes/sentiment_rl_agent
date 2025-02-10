import feedparser
import logging
import numpy as np
import json
from listennotes import podcast_api
import whisperx
from transformers import pipeline

import requests
import time
import datetime
from datetime import timezone
import os
import torch
import aiohttp
import asyncio
from typing import List, Dict, Optional
import aiofiles
from faster_whisper import WhisperModel, BatchedInferencePipeline



### STEP 1: Find Podcasts




SEARCH_KEYWORDS = ["crypto", "blockchain", "nft", "web3", "defi", "bitcoin", "ethereum", "solana", "dogecoin", "meteaverse", "polygon","shiba inu", "ripple", "cardano", "polkadot",  "chainlink", "uniswap", "binance", "coinbase","bybit", "okx", "ftx",  "binance", "coinbase", ]
ITUNES_SEARCH_URL = "https://itunes.apple.com/search"

def is_within_last_week(release_date_str: str) -> bool:
    """
    Check if the release date is within the last 7 days.
    """
    try:
        # Parse the iTunes date format (e.g., "2024-02-08T13:00:00Z")
        release_date = datetime.datetime.strptime(release_date_str, "%Y-%m-%dT%H:%M:%SZ")
        release_date = release_date.replace(tzinfo=timezone.utc)
        
        # Get current time in UTC
        now = datetime.datetime.now(timezone.utc)
        
        # Calculate the difference
        time_difference = now - release_date
        
        return time_difference.days < 7
    except (ValueError, TypeError):
        return False

async def search_single_keyword(session: aiohttp.ClientSession, 
                              keyword: str, 
                              max_results: int = 10) -> List[Dict]:
    """
    Search for podcasts for a single keyword using iTunes Search API.
    """
    try:
        logging.info(f"Searching for podcasts with keyword: {keyword}")
        
        params = {
            'term': keyword,
            'media': 'podcast',
            'entity': 'podcastEpisode',
            'limit': max_results * 3,
            'country': 'US',
            'language': 'en_us',
            'sort': 'recent'
        }
        
        headers = {
            'Accept': 'application/json'  # Explicitly request JSON response
        }
        
        async with session.get(ITUNES_SEARCH_URL, 
                             params=params, 
                             headers=headers) as response:
            if response.status != 200:
                logging.error(f"Error fetching podcasts for '{keyword}': {response.status}")
                return []
            
            # Force JSON parsing
            text = await response.text()
            data = json.loads(text)
            results = data.get('results', [])
            
            recent_episodes = []
            for item in results:
                release_date = item.get("releaseDate")
                
                if not release_date or not is_within_last_week(release_date):
                    continue
                
                episode = {
                    "title": item.get("trackName", "No Title"),
                    "description": item.get("description", "No Description"),
                    "audio_url": item.get("episodeUrl", None),
                    "podcast_name": item.get("collectionName", "Unknown Podcast"),
                    "release_date": release_date,
                    "duration": item.get("trackTimeMillis", 0),
                    "keyword_source": keyword,
                    "source": "itunes"
                }
                recent_episodes.append(episode)
                
                if len(recent_episodes) >= max_results:
                    break
            
            logging.info(f"Fetched {len(recent_episodes)} recent episodes for keyword '{keyword}'")
            return recent_episodes
            
    except Exception as e:
        logging.error(f"Error processing keyword '{keyword}': {e}")
        return []

async def search_podcasts(keywords: List[str] = SEARCH_KEYWORDS, 
                         max_results_per_keyword: int = 10) -> List[Dict]:
    """
    Search for podcasts related to multiple keywords using iTunes Search API.
    Returns a combined list of episodes from the last week.
    """
    async with aiohttp.ClientSession() as session:
        # Create tasks for each keyword search
        tasks = [
            search_single_keyword(session, keyword, max_results_per_keyword)
            for keyword in keywords
        ]
        
        # Wait for all searches to complete
        results = await asyncio.gather(*tasks)
        
        # Flatten the results list
        all_episodes = [episode for keyword_results in results for episode in keyword_results]
        
        # Remove duplicates based on audio_url
        seen_urls = set()
        unique_episodes = []
        for episode in all_episodes:
            if episode["audio_url"] and episode["audio_url"] not in seen_urls:
                seen_urls.add(episode["audio_url"])
                unique_episodes.append(episode)
        
        logging.info(f"Total unique recent episodes fetched: {len(unique_episodes)}")
        return unique_episodes




### STEP 2: Download Podcasts

async def download_podcast_audio(session: aiohttp.ClientSession,
                               episode: Dict,
                               save_dir: str = "downloads") -> Optional[str]:
    """
    Asynchronously download podcast audio file from URL.
    """
    try:
        if not episode.get("audio_url"):
            logging.error("No audio URL provided")
            return None
            
        # Create downloads directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Create filename from podcast name and episode title
        safe_filename = f"{episode['podcast_name']}_{episode['title']}"[:100]
        safe_filename = "".join(c for c in safe_filename if c.isalnum() or c in (' ', '-', '_')).strip()
        save_path = os.path.join(save_dir, f"{safe_filename}.mp3")
        
        # Skip if already downloaded
        if os.path.exists(save_path):
            logging.info(f"File already exists: {save_path}")
            return save_path
        
        async with session.get(episode["audio_url"]) as response:
            if response.status != 200:
                logging.error(f"Error downloading audio: {response.status}")
                return None
                
            async with aiofiles.open(save_path, 'wb') as f:
                while True:
                    chunk = await response.content.read(8192)
                    if not chunk:
                        break
                    await f.write(chunk)
                    
        logging.info(f"Downloaded podcast audio: {save_path}")
        return save_path
    except Exception as e:
        logging.error(f"Error downloading podcast: {e}")
        return None




### STEP 3: Transcribe Podcasts

class AsyncTranscriber:
    def __init__(self, model_size="distil-large-v3"):
        """
        Initialize the transcriber with faster-whisper model.
        Args:
            model_size: Size of the model to use (e.g., "large-v3", "distil-large-v3")
        """
        # Initialize model with GPU support and FP16 computation
        base_model = WhisperModel(
            model_size,
            device="cuda",
            compute_type="float16"
        )
        # Create batched pipeline for faster processing
        self.model = BatchedInferencePipeline(model=base_model)
        self.semaphore = asyncio.Semaphore(2)  # Limit concurrent transcriptions
    
    async def transcribe(self, audio_path: str, language: str = "en") -> Optional[Dict]:
        """
        Asynchronously transcribe podcast audio with word-level timestamps and VAD filtering.
        """
        try:
            if not os.path.exists(audio_path):
                logging.error(f"Audio file not found: {audio_path}")
                return None
            
            async with self.semaphore:
                # Run transcription in a thread pool to not block the event loop
                segments, info = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.model.transcribe(
                        audio_path,
                        language=language,
                        batch_size=16,
                        beam_size=5,
                        word_timestamps=True,
                        vad_filter=True,
                        condition_on_previous_text=False
                    )
                )
                
                # Convert generator to list to ensure completion
                segments = list(segments)
                
                # Combine all segments with timestamps
                full_text = ""
                timestamped_segments = []
                
                for segment in segments:
                    full_text += segment.text + " "
                    
                    # Add word-level details
                    words = []
                    for word in segment.words:
                        words.append({
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "probability": word.probability
                        })
                    
                    timestamped_segments.append({
                        "text": segment.text,
                        "start": segment.start,
                        "end": segment.end,
                        "words": words
                    })
                
                transcription = {
                    "text": full_text.strip(),
                    "language": info.language,
                    "language_probability": info.language_probability,
                    "segments": timestamped_segments,
                    "audio_path": audio_path
                }
                
                logging.info(f"Transcribed audio file: {audio_path}")
                logging.info(f"Detected language '{info.language}' with probability {info.language_probability}")
                
                return transcription
                
        except Exception as e:
            logging.error(f"Error transcribing podcast: {e}")
            return None




### STEP 4: Sentiment Analysis

class AsyncSentimentAnalyzer:
    def __init__(self):
        self.model = pipeline("sentiment-analysis", 
                            model="nlptown/bert-base-multilingual-uncased-sentiment")
        self.semaphore = asyncio.Semaphore(4)  # Limit concurrent analyses
    
    async def analyze_text(self, text: str) -> float:
        """
        Analyze text and return a normalized sentiment score.
        """
        if not text.strip():
            return 0.0
            
        async with self.semaphore:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.model, text
            )
            return self._normalize_score(result[0]["label"])
    
    def _normalize_score(self, label: str) -> float:
        mapping = {
            "1 star": -5.0,
            "2 stars": -2.5,
            "3 stars": 0.0,
            "4 stars": 2.5,
            "5 stars": 5.0
        }
        return mapping.get(label, 0.0)

async def process_episode(session: aiohttp.ClientSession,
                         episode: Dict,
                         transcriber: AsyncTranscriber,
                         analyzer: AsyncSentimentAnalyzer) -> Optional[Dict]:
    """
    Process a single episode through the entire pipeline asynchronously.
    """
    try:
        # Download audio
        audio_path = await download_podcast_audio(session, episode)
        if not audio_path:
            return None
            
        # Transcribe audio
        transcription = await transcriber.transcribe(audio_path)
        if not transcription:
            return None
            
        # Analyze sentiment
        sentiment_score = await analyzer.analyze_text(transcription["text"])
        
        # Combine all results
        result = {
            "episode_info": episode,
            "transcription": transcription["text"],
            "sentiment_score": sentiment_score,
            "audio_path": audio_path
        }
        
        return result
    except Exception as e:
        logging.error(f"Error processing episode: {e}")
        return None

async def save_results(results: List[Dict], filename: str = "processed_episodes.json"):
    """
    Asynchronously save results to JSON file.
    """
    try:
        current_time = datetime.datetime.now().isoformat()
        async with aiofiles.open(filename, 'w', encoding='utf-8') as f:
            await f.write(json.dumps({
                "episodes": results,
                "total_episodes": len(results),
                "timestamp": current_time
            }, indent=2, ensure_ascii=False))
        logging.info(f"Successfully saved {len(results)} episodes to {filename}")
    except Exception as e:
        logging.error(f"Error saving results: {e}")




### STEP 5: Aggregate Sentiment
def get_podcast_sentiment(episodes):
    """
    Fetch latest podcast episodes, transcribe, and analyze sentiment.
    """
    abs_analyzer = AsyncSentimentAnalyzer()
    all_transcripts = []
    
    for episode in episodes:
        logging.info(f"Processing podcast: {episode['title']}")
        if episode.get('audio_url'):
            audio_path = download_podcast_audio(episode)
            if audio_path:
                transcript = transcribe_podcast(audio_path)
                if transcript:
                    all_transcripts.append(transcript)
    
    # Run sentiment analysis on all transcripts
    sentiment_scores = [analyze_sentiment(transcript)["sentiment_score"] 
                       for transcript in all_transcripts if transcript]
    return np.mean(sentiment_scores) if sentiment_scores else 0.0




def save_episodes_to_json(listen_notes_episodes,  filename="podcast_episodes.json"):
    """
    Save all podcast episodes to a JSON file.
    """
    all_episodes = {
        "listen_notes_episodes": listen_notes_episodes,
        "total_episodes": len(listen_notes_episodes)
    }
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(all_episodes, f, indent=2, ensure_ascii=False)
        logging.info(f"Successfully saved {all_episodes['total_episodes']} episodes to {filename}")
    except Exception as e:
        logging.error(f"Error saving episodes to JSON: {e}")


def save_podcast_sentiment(podcast_sentiment, filename="podcast_sentiment.json"):
    """
    Save the podcast sentiment to a JSON file.
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({"sentiment": podcast_sentiment}, f, indent=2, ensure_ascii=False)
        logging.info(f"Successfully saved podcast sentiment to {filename}")
    except Exception as e:
        logging.error(f"Error saving podcast sentiment: {e}")





##### MAIN EXECUTION

async def main():
    try:
        logging.basicConfig(level=logging.INFO)
        logging.info("=== Starting the Async Podcast Collection Pipeline ===")
        
        # Initialize components
        transcriber = AsyncTranscriber()
        analyzer = AsyncSentimentAnalyzer()
        
        async with aiohttp.ClientSession() as session:
            # Step 1: Search for podcasts
            episodes = await search_podcasts(SEARCH_KEYWORDS)
            logging.info(f"Found {len(episodes)} episodes")
            
            # Step 2: Process episodes concurrently
            tasks = [
                process_episode(session, episode, transcriber, analyzer)
                for episode in episodes
            ]
            results = await asyncio.gather(*tasks)
            results = [r for r in results if r is not None]
            
            # Step 3: Save results
            await save_results(results)
            
            logging.info(f"Successfully processed {len(results)} episodes")
            logging.info("=== Podcast Collection Complete ===")
            
    except Exception as e:
        logging.error(f"An error occurred during execution: {e}")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())
