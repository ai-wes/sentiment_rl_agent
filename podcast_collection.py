import feedparser
import logging
import numpy as np
import json
import whisperx
from transformers import pipeline
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
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
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter



### STEP 1: Find Podcasts
device = "cuda" if torch.cuda.is_available() else "cpu"


SEARCH_KEYWORDS = ["crypto", "blockchain", "nft", "web3", "defi", "bitcoin", "ethereum", "solana", "dogecoin", "meteaverse", "polygon","shiba inu", "ripple", "cardano", "polkadot",  "chainlink", "uniswap", "binance", "coinbase","bybit", "okx", "ftx",  "binance", "coinbase", ]
ITUNES_SEARCH_URL = "https://itunes.apple.com/search"




torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)    
processor = AutoProcessor.from_pretrained(model_id)        
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    return_timestamps=True
)





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

def transcribe_podcast(audio_path: str):
    dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
    sample = dataset[0]["audio"]


    result = pipe(sample)
    print(result["text"])        
    # Initialize model with appropriate device and compute type

    logging.info(f"Model loaded successfully")

    # Create batched pipeline for faster processing
    semaphore = asyncio.Semaphore(2)

    # Create transcripts directory
    transcripts_dir = "transcripts"


    os.makedirs(transcripts_dir, exist_ok=True)

    return result["text"]



async def save_transcription(transcription: Dict, audio_path: str, transcripts_dir: str = "transcripts") -> str:
    """
    Save transcription to a JSON file.
    """
    try:

        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        safe_name = "".join(c for c in base_name if c.isalnum() or c in (' ', '-', '_')).strip()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_name}_{timestamp}.json"
        filepath = os.path.join(transcripts_dir, filename)


        transcription_with_metadata = {
            "metadata": {
                "timestamp": datetime.datetime.now().isoformat(),
                "audio_file": audio_path,
            },
            "transcription": transcription

        }

        async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(transcription_with_metadata, indent=2, ensure_ascii=False))
        
        logging.info(f"Saved transcription to: {filepath}")
        return filepath

    except Exception as e:
        logging.error(f"Error saving transcription for {audio_path}: {str(e)}")
        return ""





### STEP 4: Sentiment Analysis

class AsyncSentimentAnalyzer:
    def __init__(self, device=None):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Initializing sentiment analyzer on device: {self.device}")
        
        self.model = pipeline("sentiment-analysis", 
                            model="nlptown/bert-base-multilingual-uncased-sentiment",
                            device=self.device)
        self.semaphore = asyncio.Semaphore(4)
        self.max_length = 450
        logging.info("Initialized AsyncSentimentAnalyzer")
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks that fit within model's token limit.
        More aggressive chunking to prevent token overflow.
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word.split())
            
            if current_length + word_length > self.max_length:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [word]
                    current_length = word_length
                else:
                    # Handle very long words by forcing a split
                    chunks.append(word[:self.max_length])
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        logging.info(f"Split text into {len(chunks)} chunks for analysis")
        for i, chunk in enumerate(chunks, 1):
            logging.info(f"Chunk {i} length (words): {len(chunk.split())}")
        
        return chunks
    
    async def analyze_text(self, text: str) -> float:
        """
        Analyze text and return a normalized sentiment score.
        """
        if not text.strip():
            logging.warning("Received empty text for sentiment analysis")
            return 0.0
        
        try:
            chunks = self._chunk_text(text)
            scores = []
            
            async with self.semaphore:
                for i, chunk in enumerate(chunks):
                    logging.info(f"Analyzing chunk {i+1}/{len(chunks)}")
                    try:
                        result = await asyncio.get_event_loop().run_in_executor(
                            None, self.model, chunk
                        )
                        chunk_score = self._normalize_score(result[0]["label"])
                        scores.append(chunk_score)
                        logging.info(f"Chunk {i+1} sentiment score: {chunk_score}")
                    except Exception as e:
                        logging.error(f"Error analyzing chunk {i+1}: {str(e)}")
                        continue
            
            final_score = sum(scores) / len(scores) if scores else 0.0
            logging.info(f"Final averaged sentiment score: {final_score}")
            return final_score
            
        except Exception as e:
            logging.error(f"Error in sentiment analysis: {str(e)}")
            return 0.0
    
    def _normalize_score(self, label: str) -> float:
        """
        Convert star ratings to normalized sentiment scores.
        """
        mapping = {
            "1 star": -5.0,
            "2 stars": -2.5,
            "3 stars": 0.0,
            "4 stars": 2.5,
            "5 stars": 5.0
        }
        score = mapping.get(label, 0.0)
        logging.debug(f"Normalized sentiment score: {label} -> {score}")
        return score

async def process_episode(session: aiohttp.ClientSession,
                         episode: Dict,
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
        transcription = await transcribe_podcast(audio_path)
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



analyzer = AsyncSentimentAnalyzer()


### STEP 5: Aggregate Sentiment
def get_podcast_sentiment(episodes):

    """
    Fetch latest podcast episodes, transcribe, and analyze sentiment.
    """
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
    sentiment_scores = [analyzer.analyze_text(transcript)["sentiment_score"] 
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

class PodcastTranscriber:
    def __init__(self, model_size="distil-large-v3", transcripts_dir="transcripts", device=None):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_size = model_size
        logging.info(f"Initializing PodcastTranscriber with model size: {model_size} on device: {self.device}")
        
        # Set compute type based on device
        compute_type = "float16" if self.device.startswith("cuda") else "float32"
        
        try:
            # Initialize model with appropriate device and compute type
            self.model = WhisperModel(
                model_size,
                device=self.device,
                compute_type=compute_type
            )
            
            # Create transcripts directory
            self.transcripts_dir = transcripts_dir
            os.makedirs(self.transcripts_dir, exist_ok=True)
            
            logging.info("Successfully initialized PodcastTranscriber")
            
        except Exception as e:
            logging.error(f"Error initializing WhisperModel: {str(e)}")
            raise

    def transcribe(self, audio_path: str) -> Optional[Dict]:
        """
        Transcribe audio file using WhisperModel
        """
        try:
            logging.info(f"Starting transcription of: {audio_path}")
            
            # Run transcription without the unsupported 'logprob_threshold' parameter
            segments, info = self.model.transcribe(
                audio_path,
                beam_size=5,
                word_timestamps=True,
                initial_prompt="This is a podcast about cryptocurrency and blockchain technology.",
                condition_on_previous_text=True,
                temperature=0.0,
                compression_ratio_threshold=2.4,
                no_speech_threshold=0.6,
                vad_filter=True
            )

            # Convert generator to list and process segments
            segments_list = list(segments)
            
            # Process segments and words
            processed_segments = []
            full_text = ""
            
            for segment in segments_list:
                # Process words in segment
                words = []
                if hasattr(segment, 'words'):
                    for word in segment.words:
                        words.append({
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "probability": word.probability
                        })
                
                processed_segments.append({
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end,
                    "words": words
                })
                
                full_text += segment.text + " "
            
            transcription = {
                "text": full_text.strip(),
                "language": info.language,
                "language_probability": info.language_probability,
                "segments": processed_segments
            }
            
            logging.info(f"Completed transcription of: {audio_path}")
            return transcription
            
        except Exception as e:
            logging.error(f"Error transcribing {audio_path}: {str(e)}", exc_info=True)
            return None

    def save_transcription(self, transcription: Dict, audio_path: str) -> str:
        """
        Save transcription to a JSON file
        """
        try:
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            safe_name = "".join(c for c in base_name if c.isalnum() or c in (' ', '-', '_')).strip()
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{safe_name}_{timestamp}.json"
            filepath = os.path.join(self.transcripts_dir, filename)

            transcription_with_metadata = {
                "metadata": {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "audio_file": audio_path,
                    "model_size": self.model_size
                },
                "transcription": transcription
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(transcription_with_metadata, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Saved transcription to: {filepath}")
            return filepath

        except Exception as e:
            logging.error(f"Error saving transcription for {audio_path}: {str(e)}")
            return ""

    def process_audio_files(self, audio_files: List[str]) -> List[str]:
        """
        Process multiple audio files sequentially
        """
        transcript_paths = []
        logging.info(f"Starting batch transcription of {len(audio_files)} files")
        
        for audio_file in audio_files:
            try:
                logging.info(f"Processing: {audio_file}")
                transcript = self.transcribe(audio_file)
                if transcript:
                    save_path = self.save_transcription(transcript, audio_file)
                    if save_path:
                        transcript_paths.append(save_path)
                        logging.info(f"Successfully processed: {audio_file}")
            except Exception as e:
                logging.error(f"Error processing {audio_file}: {str(e)}")
                continue
        
        return transcript_paths

def main():
    try:
        # Initialize transcriber
        transcriber = PodcastTranscriber()
        
        # Get list of audio files
        audio_files = [f for f in os.listdir("downloads") if f.endswith(".mp3")]
        audio_paths = [os.path.join("downloads", f) for f in audio_files]
        
        # Process all audio files
        transcript_paths = transcriber.process_audio_files(audio_paths)
        
        logging.info(f"Completed processing {len(transcript_paths)} files")
        
    except Exception as e:
        logging.error(f"Error in main process: {e}")
        raise

if __name__ == "__main__":
    main()


