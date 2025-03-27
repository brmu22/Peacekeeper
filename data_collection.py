import os
import json
import pandas as pd
import logging
import requests
from datetime import datetime
import time
import random
import re

logger = logging.getLogger(__name__)

class DataCollectionOrchestrator:
    """
    Orchestrates the collection of stress-related data from multiple sources.
    """
    
    def __init__(self, config_path):
        """
        Initialize the data collection orchestrator.
        
        Args:
            config_path (str): Path to the configuration file.
        """
        self.config_path = config_path
        self.load_config()
        
        # Create data directory if it doesn't exist
        os.makedirs(self.config.get("data_path", "data/"), exist_ok=True)
        
        # Initialize data collectors
        self.collectors = {
            "reddit": RedditDataCollector(self.config),
            "twitter": TwitterDataCollector(self.config),
            "survey": SurveyDataCollector(self.config)
        }
    
    def load_config(self):
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            logger.info("Config loaded successfully")
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            self.config = {}
    
    def collect_all_data(self):
        """
        Collect data from all available sources.
        """
        all_data = []
        
        for source, collector in self.collectors.items():
            logger.info(f"Collecting data from {source}...")
            try:
                source_data = collector.collect_data()
                if source_data is not None and not source_data.empty:
                    # Add source column
                    source_data['source'] = source
                    all_data.append(source_data)
                    logger.info(f"Collected {len(source_data)} records from {source}")
                else:
                    logger.warning(f"No data collected from {source}")
            except Exception as e:
                logger.error(f"Error collecting data from {source}: {str(e)}")
        
        # Combine all data
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            logger.info(f"Total records collected: {len(combined_data)}")
            
            # Save combined data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                self.config.get("data_path", "data/"),
                f"collected_data_{timestamp}.csv"
            )
            combined_data.to_csv(output_path, index=False)
            logger.info(f"Combined data saved to {output_path}")
            
            return combined_data
        else:
            logger.warning("No data collected from any source")
            return pd.DataFrame()


class BaseDataCollector:
    """
    Base class for data collectors.
    """
    
    def __init__(self, config):
        """
        Initialize the data collector.
        
        Args:
            config (dict): Configuration dictionary.
        """
        self.config = config
    
    def collect_data(self):
        """
        Collect data from the source.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement collect_data method")
    
    def process_data(self, raw_data):
        """
        Process the collected raw data.
        
        Args:
            raw_data: Raw data collected from the source.
            
        Returns:
            pd.DataFrame: Processed data.
        """
        raise NotImplementedError("Subclasses must implement process_data method")
    
    def save_data(self, data, source_name):
        """
        Save the collected data.
        
        Args:
            data (pd.DataFrame): Collected data.
            source_name (str): Name of the data source.
        """
        if data is None or data.empty:
            logger.warning(f"No data to save for {source_name}")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            self.config.get("data_path", "data/"),
            f"{source_name}_data_{timestamp}.csv"
        )
        data.to_csv(output_path, index=False)
        logger.info(f"Data from {source_name} saved to {output_path}")
        return output_path


class RedditDataCollector(BaseDataCollector):
    """
    Collector for Reddit data.
    """
    
    def collect_data(self):
        """
        Collect stress-related data from Reddit.
        
        Returns:
            pd.DataFrame: Collected data.
        """
        logger.info("Collecting data from Reddit")
        
        # Check if API credentials are available
        api_config = self.config.get("reddit_api", {})
        if not api_config.get("client_id") or not api_config.get("client_secret"):
            logger.warning("Reddit API credentials not found, using simulated data")
            return self._simulate_reddit_data()
        
        # Implement actual Reddit API collection here
        # For this example, we'll use simulated data
        return self._simulate_reddit_data()
    
    def _simulate_reddit_data(self):
        """
        Simulate Reddit data for testing purposes.
        
        Returns:
            pd.DataFrame: Simulated Reddit data.
        """
        subreddits = ["stress", "anxiety", "mentalhealth", "depression", "work"]
        stress_keywords = ["stressed", "overwhelmed", "anxious", "pressure", "burnout"]
        
        records = []
        
        # Generate simulated posts
        for _ in range(50):
            subreddit = random.choice(subreddits)
            keyword = random.choice(stress_keywords)
            
            # Generate text with varying stress levels
            stress_level = random.choice(["low", "medium", "high"])
            
            if stress_level == "low":
                text = f"I've been dealing with some {keyword} feelings lately, but it's manageable. "
                text += "I've been taking care of myself and things are getting better."
            elif stress_level == "medium":
                text = f"This {keyword} situation at work/school is getting to me. "
                text += "I'm finding it harder to cope and it's affecting my sleep and mood."
            else:  # high
                text = f"I'm completely {keyword} and can't handle it anymore. "
                text += "Everything feels like it's falling apart and I don't know what to do."
            
            # Add random variation
            variations = [
                "Anyone else feel this way?",
                "How do you deal with this?",
                "Any advice would be appreciated.",
                "Thanks for listening.",
                "I just needed to vent."
            ]
            text += " " + random.choice(variations)
            
            # Create record
            timestamp = datetime.now().timestamp() - random.randint(0, 30*24*60*60)  # Up to 30 days ago
            record = {
                "text": text,
                "subreddit": subreddit,
                "timestamp": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                "upvotes": random.randint(1, 100),
                "num_comments": random.randint(0, 20),
                "stress_level": stress_level  # Ground truth for testing
            }
            records.append(record)
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        # Save the data
        self.save_data(df, "reddit")
        
        return df


class TwitterDataCollector(BaseDataCollector):
    """
    Collector for Twitter data.
    """
    
    def collect_data(self):
        """
        Collect stress-related data from Twitter.
        
        Returns:
            pd.DataFrame: Collected data.
        """
        logger.info("Collecting data from Twitter")
        
        # Check if API credentials are available
        api_config = self.config.get("twitter_api", {})
        if not api_config.get("api_key") or not api_config.get("api_secret"):
            logger.warning("Twitter API credentials not found, using simulated data")
            return self._simulate_twitter_data()
        
        # Implement actual Twitter API collection here
        # For this example, we'll use simulated data
        return self._simulate_twitter_data()
    
    def _simulate_twitter_data(self):
        """
        Simulate Twitter data for testing purposes.
        
        Returns:
            pd.DataFrame: Simulated Twitter data.
        """
        hashtags = ["#stress", "#mentalhealth", "#anxiety", "#burnout", "#worklife"]
        stress_phrases = [
            "feeling overwhelmed",
            "need a break",
            "too much pressure",
            "can't handle this",
            "stressed out",
            "exhausted",
            "mental health day",
            "work-life balance"
        ]
        
        records = []
        
        # Generate simulated tweets
        for _ in range(50):
            # Pick random hashtags
            tweet_hashtags = random.sample(hashtags, k=random.randint(1, 3))
            phrase = random.choice(stress_phrases)
            
            # Generate text with varying stress levels
            stress_level = random.choice(["low", "medium", "high"])
            
            if stress_level == "low":
                text = f"Slightly {phrase} today, but I'll manage. {' '.join(tweet_hashtags)}"
            elif stress_level == "medium":
                text = f"Really {phrase} with everything going on. Need to find better coping strategies. {' '.join(tweet_hashtags)}"
            else:  # high
                text = f"Completely {phrase} and at my breaking point! I can't continue like this. {' '.join(tweet_hashtags)}"
            
            # Create record
            timestamp = datetime.now().timestamp() - random.randint(0, 7*24*60*60)  # Up to 7 days ago
            record = {
                "text": text,
                "hashtags": ",".join(tweet_hashtags),
                "timestamp": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                "likes": random.randint(0, 50),
                "retweets": random.randint(0, 10),
                "stress_level": stress_level  # Ground truth for testing
            }
            records.append(record)
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        # Save the data
        self.save_data(df, "twitter")
        
        return df


class SurveyDataCollector(BaseDataCollector):
    """
    Collector for survey data.
    """
    
    def collect_data(self):
        """
        Collect stress-related data from surveys.
        
        Returns:
            pd.DataFrame: Collected data.
        """
        logger.info("Collecting data from surveys")
        
        # Check if survey data path is available
        survey_path = self.config.get("survey_data_path", "")
        if survey_path and os.path.exists(survey_path):
            logger.info(f"Loading survey data from {survey_path}")
            try:
                survey_data = pd.read_csv(survey_path)
                return self.process_data(survey_data)
            except Exception as e:
                logger.error(f"Error loading survey data: {str(e)}")
        
        logger.warning("Survey data path not found or invalid, using simulated data")
        return self._simulate_survey_data()
    
    def process_data(self, raw_data):
        """
        Process the raw survey data.
        
        Args:
            raw_data (pd.DataFrame): Raw survey data.
            
        Returns:
            pd.DataFrame: Processed survey data.
        """
        # Implement processing for actual survey data here
        return raw_data
    
    def _simulate_survey_data(self):
        """
        Simulate survey data for testing purposes.
        
        Returns:
            pd.DataFrame: Simulated survey data.
        """
        stress_questions = [
            "How would you describe your current stress level?",
            "Please describe what's causing you stress:",
            "How are you coping with your stress?",
            "What impacts is stress having on your life?",
            "What support would help you manage stress better?"
        ]
        
        records = []
        
        # Generate simulated responses
        for _ in range(50):
            # Randomly select which question to answer
            question = random.choice(stress_questions)
            
            # Generate response with varying stress levels
            stress_level = random.choice(["low", "medium", "high"])
            
            if stress_level == "low":
                if "current stress level" in question:
                    response = "My stress level is fairly low right now. I have some minor concerns but nothing major."
                elif "causing" in question:
                    response = "Some minor work deadlines and household responsibilities, but nothing too serious."
                elif "coping" in question:
                    response = "I'm taking regular breaks, exercising, and talking with friends when needed."
                elif "impacts" in question:
                    response = "Minimal impact. Sometimes I feel a bit tired but generally I'm doing well."
                else:  # support
                    response = "I have good support already, but maybe some more organizational tips would help."
            
            elif stress_level == "medium":
                if "current stress level" in question:
                    response = "My stress level is moderate. I'm managing but it's definitely noticeable."
                elif "causing" in question:
                    response = "Work pressure is building up, plus I have family responsibilities and financial concerns."
                elif "coping" in question:
                    response = "I try to manage with exercise and relaxation techniques, but it's not always effective."
                elif "impacts" in question:
                    response = "It's affecting my sleep sometimes and I feel more irritable than usual."
                else:  # support
                    response = "Better time management strategies and perhaps some counseling would help."
            
            else:  # high
                if "current stress level" in question:
                    response = "My stress level is extremely high. I feel overwhelmed almost constantly."
                elif "causing" in question:
                    response = "Everything - work deadlines, financial problems, relationship issues, health concerns."
                elif "coping" in question:
                    response = "I'm struggling to cope. My usual strategies aren't working and I feel burnt out."
                elif "impacts" in question:
                    response = "It's seriously affecting my sleep, appetite, mood, and ability to function at work and home."
                else:  # support
                    response = "I need professional help, possibly medication, and significant lifestyle changes."
            
            # Create record
            timestamp = datetime.now().timestamp() - random.randint(0, 90*24*60*60)  # Up to 90 days ago
            record = {
                "text": response,
                "question": question,
                "timestamp": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                "age_group": random.choice(["18-24", "25-34", "35-44", "45-54", "55+"]),
                "gender": random.choice(["Male", "Female", "Non-binary", "Prefer not to say"]),
                "stress_level": stress_level  # Ground truth for testing
            }
            records.append(record)
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        # Save the data
        self.save_data(df, "survey")
        
        return df


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test the data collection
    orchestrator = DataCollectionOrchestrator("config.json")
    collected_data = orchestrator.collect_all_data()
    
    if not collected_data.empty:
        print(f"Collected {len(collected_data)} records")
        print("\nSample data:")
        print(collected_data.head())
    else:
        print("No data collected")