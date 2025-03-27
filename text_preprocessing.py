import os
import json
import pandas as pd
import numpy as np
import re
import string
import logging
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

# Try to download NLTK resources, handle offline scenario
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except:
    logging.warning("NLTK download failed. If you're offline, this is expected.")

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """
    Class for preprocessing text data and extracting features.
    """
    
    def __init__(self, config=None):
        """
        Initialize the text preprocessor.
        
        Args:
            config (dict, optional): Configuration dictionary. Defaults to None.
        """
        self.config = config or {}
        
        # Initialize NLTK components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        try:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        except:
            self.sentiment_analyzer = None
            logger.warning("SentimentIntensityAnalyzer not available. Sentiment scores will not be computed.")
        
        # Load stress keywords and thresholds
        self.stress_keywords = {
            'high': [
                'overwhelmed', 'desperate', 'unbearable', 'exhausted', 'breakdown',
                'panic', 'anxiety', 'depressed', 'hopeless', 'crisis', 'severe',
                'extreme', 'terrible', 'awful', 'cannot', 'impossible', 'never',
                'worst', 'suicidal', 'terrified'
            ],
            'medium': [
                'stressed', 'worried', 'concerned', 'upset', 'struggling',
                'difficult', 'hard', 'trouble', 'pressure', 'tension',
                'anxious', 'nervous', 'fear', 'tired', 'problem', 'challenge',
                'burden', 'strain', 'uncomfortable'
            ],
            'low': [
                'fine', 'okay', 'alright', 'calm', 'relaxed', 'peaceful',
                'balanced', 'coping', 'managing', 'handling', 'contained',
                'controlled', 'mild', 'minimal', 'little', 'slight', 'rare',
                'occasionally', 'sometimes', 'minor'
            ]
        }
        
        self.stress_thresholds = self.config.get('stress_thresholds', {
            'low': [-1.0, -0.3],
            'medium': [-0.3, 0.1],
            'high': [0.1, 1.0]
        })
    
    def process_text(self, text):
        """
        Process a single text sample.
        
        Args:
            text (str): The text to process.
            
        Returns:
            dict: Processed features.
        """
        if not text or not isinstance(text, str):
            return {
                'cleaned_text': '',
                'tokens': [],
                'vader_scores': {'compound': 0, 'neg': 0, 'neu': 0, 'pos': 0},
                'stress_level': 'medium',  # Default to medium if text is invalid
                'stress_keywords_found': {}
            }
        
        # Clean text
        cleaned_text = self._clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(cleaned_text.lower())
        
        # Remove stopwords and lemmatize
        processed_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and token not in string.punctuation
        ]
        
        # Get sentiment scores
        vader_scores = self._get_sentiment_scores(text)
        
        # Count stress keywords
        stress_keywords_found = self._count_stress_keywords(processed_tokens)
        
        # Determine stress level
        stress_level = self._determine_stress_level(vader_scores['compound'], stress_keywords_found)
        
        return {
            'cleaned_text': cleaned_text,
            'tokens': processed_tokens,
            'vader_scores': vader_scores,
            'stress_level': stress_level,
            'stress_keywords_found': stress_keywords_found
        }
    
    def _clean_text(self, text):
        """
        Clean and normalize text.
        
        Args:
            text (str): Text to clean.
            
        Returns:
            str: Cleaned text.
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _get_sentiment_scores(self, text):
        """
        Get sentiment scores for text.
        
        Args:
            text (str): Text to analyze.
            
        Returns:
            dict: Sentiment scores.
        """
        if self.sentiment_analyzer:
            return self.sentiment_analyzer.polarity_scores(text)
        else:
            return {'compound': 0, 'neg': 0, 'neu': 0, 'pos': 0}
    
    def _count_stress_keywords(self, tokens):
        """
        Count stress keywords in tokens.
        
        Args:
            tokens (list): List of tokens.
            
        Returns:
            dict: Counts of stress keywords by level.
        """
        counts = {level: 0 for level in self.stress_keywords.keys()}
        
        for token in tokens:
            for level, keywords in self.stress_keywords.items():
                if token in keywords:
                    counts[level] += 1
        
        return counts
    
    def _determine_stress_level(self, sentiment_score, keyword_counts):
        """
        Determine stress level based on sentiment and keywords.
        
        Args:
            sentiment_score (float): Sentiment score.
            keyword_counts (dict): Counts of stress keywords.
            
        Returns:
            str: Stress level (low, medium, high).
        """
        # Get total keyword counts
        total_keywords = sum(keyword_counts.values())
        
        if total_keywords == 0:
            # Use only sentiment if no keywords found
            for level, (min_val, max_val) in self.stress_thresholds.items():
                if min_val <= sentiment_score <= max_val:
                    return level
            return 'medium'  # Default to medium
        
        # Calculate weighted score
        # Sentiment contributes 60%, keywords 40%
        normalized_keyword_scores = {
            level: count / total_keywords if total_keywords > 0 else 0 
            for level, count in keyword_counts.items()
        }
        
        score = 0.0
        # Convert sentiment to a -1 to 1 scale where -1 is low stress, 1 is high stress
        # This is the opposite of VADER's compound score (where negative is negative sentiment)
        sentiment_weight = -sentiment_score  # Invert because negative sentiment means higher stress
        
        # Weight the keyword contributions
        keyword_weight = (
            normalized_keyword_scores['high'] * 0.8 + 
            normalized_keyword_scores['medium'] * 0.0 + 
            normalized_keyword_scores['low'] * -0.8
        )
        
        # Combine scores
        combined_score = 0.6 * sentiment_weight + 0.4 * keyword_weight
        
        # Determine level based on combined score
        if combined_score < -0.3:
            return 'low'
        elif combined_score > 0.1:
            return 'high'
        else:
            return 'medium'


class DataPreprocessor:
    """
    Class for preprocessing datasets.
    """
    
    def __init__(self, config_path):
        """
        Initialize the data preprocessor.
        
        Args:
            config_path (str): Path to configuration file.
        """
        self.config_path = config_path
        self.load_config()
        
        # Create directories if they don't exist
        self.data_path = self.config.get('data_path', 'data/')
        self.processed_path = self.config.get('processed_data_path', 'processed_data/')
        
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.processed_path, exist_ok=True)
        
        # Initialize text preprocessor
        self.text_preprocessor = TextPreprocessor(self.config)
    
    def load_config(self):
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            logger.info("Config loaded successfully")
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            self.config = {}
    
    def process_all_files(self):
        """
        Process all data files in the data directory.
        """
        # Get list of CSV files in data directory
        data_files = [f for f in os.listdir(self.data_path) if f.endswith('.csv')]
        
        if not data_files:
            logger.warning(f"No CSV files found in {self.data_path}")
            return
        
        logger.info(f"Found {len(data_files)} CSV files to process")
        
        for file in data_files:
            input_path = os.path.join(self.data_path, file)
            self.process_file(input_path)
    
    def process_file(self, file_path):
        """
        Process a single data file.
        
        Args:
            file_path (str): Path to the data file.
        """
        logger.info(f"Processing file: {file_path}")
        
        try:
            # Load data
            df = pd.read_csv(file_path)
            
            # Check if 'text' column exists
            if 'text' not in df.columns:
                logger.error(f"No 'text' column found in {file_path}")
                return
            
            # Process text
            processed_data = []
            for _, row in df.iterrows():
                text = row['text']
                processed = self.text_preprocessor.process_text(text)
                
                # Combine with original data
                processed_row = {**row.to_dict(), **processed}
                processed_data.append(processed_row)
            
            # Create DataFrame
            processed_df = pd.DataFrame(processed_data)
            
            # Save processed data
            output_filename = f"processed_{os.path.basename(file_path)}"
            output_path = os.path.join(self.processed_path, output_filename)
            processed_df.to_csv(output_path, index=False)
            
            logger.info(f"Processed data saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
    
    def prepare_training_data(self, output_dir='processed_data'):
        """
        Prepare training and validation datasets from processed data.
        
        Args:
            output_dir (str, optional): Directory to save the datasets. Defaults to 'processed_data'.
            
        Returns:
            tuple: Training and validation DataFrames.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get list of processed files
        processed_files = [
            f for f in os.listdir(self.processed_path) 
            if f.startswith('processed_') and f.endswith('.csv')
        ]
        
        if not processed_files:
            logger.warning(f"No processed files found in {self.processed_path}")
            return None, None
        
        # Combine all processed data
        all_data = []
        for file in processed_files:
            file_path = os.path.join(self.processed_path, file)
            df = pd.read_csv(file_path)
            
            # Ensure required columns exist
            if 'text' not in df.columns or 'stress_level' not in df.columns:
                logger.warning(f"Required columns missing in {file_path}")
                continue
            
            all_data.append(df)
        
        if not all_data:
            logger.warning("No valid processed data found")
            return None, None
        
        # Combine data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Remove duplicates
        combined_df = combined_df.drop_duplicates(subset=['text'])
        
        # Split into training and validation sets
        train_df, val_df = train_test_split(
            combined_df,
            test_size=self.config.get('training', {}).get('validation_split', 0.2),
            random_state=42,
            stratify=combined_df['stress_level'] if 'stress_level' in combined_df.columns else None
        )
        
        # Save datasets
        train_path = os.path.join(output_dir, 'training_data.csv')
        val_path = os.path.join(output_dir, 'validation_data.csv')
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        
        logger.info(f"Training data ({len(train_df)} samples) saved to {train_path}")
        logger.info(f"Validation data ({len(val_df)} samples) saved to {val_path}")
        
        return train_df, val_df


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test the text preprocessor
    text_preprocessor = TextPreprocessor()
    
    sample_texts = [
        "I'm feeling great today. Everything is going smoothly and I'm enjoying my work.",
        "It's been a busy day but I'm managing fine. Some minor concerns but nothing serious.",
        "I'm extremely stressed about this project deadline. The pressure is overwhelming and I can't sleep well."
    ]
    
    print("Testing TextPreprocessor:")
    for text in sample_texts:
        result = text_preprocessor.process_text(text)
        print(f"\nText: {text}")
        print(f"Stress Level: {result['stress_level']}")
        print(f"Sentiment Score: {result['vader_scores']['compound']:.4f}")
        print(f"Keywords found: {result['stress_keywords_found']}")
    
    # Test data preprocessor if config file exists
    if os.path.exists("config.json"):
        print("\nTesting DataPreprocessor:")
        data_preprocessor = DataPreprocessor("config.json")
        data_preprocessor.process_all_files()
        train_df, val_df = data_preprocessor.prepare_training_data()
        
        if train_df is not None and val_df is not None:
            print(f"Training data: {len(train_df)} samples")
            print(f"Validation data: {len(val_df)} samples")
    
