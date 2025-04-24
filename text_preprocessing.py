import os
import json
import pandas as pd
import numpy as np
import re
import logging
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

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

# Negation and Emoji sentiment mappings
NEGATION_WORDS = {"not", "no", "never", "none", "cannot", "can't", "won't", "don't"}
EMOJI_SENTIMENT = {
    "😊": 0.5, "😔": -0.5, "😢": -0.6,
    "😃": 0.6, "😡": -0.7, "😞": -0.6, "😠": -0.7
}

class TextPreprocessor:
    """
    Class for preprocessing text data and extracting features.
    Enhanced with negation handling, emoji sentiment, and refined scoring.
    """
    def __init__(self, config=None):
        self.config = config or {}
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        try:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        except:
            self.sentiment_analyzer = None
            logger.warning("SentimentIntensityAnalyzer not available. Sentiment scores default to 0.")

        # Load stress keywords and thresholds
        default_thresholds = self.config.get('stress_thresholds', {
            'low': [-1.0, -0.3],
            'medium': [-0.3, 0.1],
            'high': [0.1, 1.0]
        })
        self.stress_thresholds = default_thresholds
        self.stress_keywords = self.config.get('stress_keywords', {
            'high': [
                'overwhelmed', 'desperate', 'unbearable', 'exhausted', 'breakdown',
                'panic', 'anxiety', 'depressed', 'hopeless', 'crisis', 'severe',
                'extreme', 'terrible', 'awful', 'cannot', 'impossible', 'never',
                'worst', 'suicidal', 'terrified', 'burnout', 'stressful'
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
        })

    def process_text(self, text):
        if not text or not isinstance(text, str):
            return {
                'cleaned_text': '',
                'tokens': [],
                'vader_scores': {'compound': 0, 'neg': 0, 'neu': 0, 'pos': 0},
                'stress_level': 'medium',
                'stress_keywords_found': {}
            }

        # Clean text (preserve emojis)
        cleaned_text = self._clean_text(text)
        # Tokenize and normalize
        tokens = word_tokenize(cleaned_text)
        tokens = [self.lemmatizer.lemmatize(t.lower()) for t in tokens
                  if t.lower() not in self.stop_words]

        # Sentiment (VADER + emojis)
        vader = self._get_sentiment_scores(text)
        emoji_score = sum(EMOJI_SENTIMENT.get(ch, 0) for ch in text)
        compound = vader['compound'] if self.sentiment_analyzer else 0
        # combine: 80% vader, 20% emoji, clamp to [-1,1]
        combined_compound = np.clip(0.8 * compound + 0.2 * np.tanh(emoji_score), -1, 1)
        vader['compound'] = combined_compound

        # Keyword counts with negation handling
        keyword_counts = {lvl: 0 for lvl in self.stress_keywords}
        for i, token in enumerate(tokens):
            for level, kw_list in self.stress_keywords.items():
                if token in kw_list:
                    # skip if preceded by negation
                    window = tokens[max(i-3, 0):i]
                    if any(w in NEGATION_WORDS for w in window):
                        continue
                    keyword_counts[level] += 1

        # Determine stress level
        stress_level = self._determine_stress_level(combined_compound, keyword_counts)

        return {
            'cleaned_text': cleaned_text,
            'tokens': tokens,
            'vader_scores': vader,
            'stress_level': stress_level,
            'stress_keywords_found': keyword_counts
        }

    def _clean_text(self, text):
        text = text.lower()
        # preserve emojis, remove URLs & tags
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        # remove unwanted punctuation but keep emojis
        text = re.sub(r'["#$%&\'()*+,\-./:;<=>?@\[\]^_`{|}~]', '', text)
        # remove digits
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _get_sentiment_scores(self, text):
        if self.sentiment_analyzer:
            return self.sentiment_analyzer.polarity_scores(text)
        return {'compound': 0, 'neg': 0, 'neu': 0, 'pos': 0}

    def _determine_stress_level(self, sentiment_score, keyword_counts):
        total = sum(keyword_counts.values())
        # normalize keyword score
        norm = {lvl: cnt / total if total else 0 for lvl, cnt in keyword_counts.items()}
        # sentiment weight inverted: negative sentiment → high stress
        sent_w = -sentiment_score
        # keyword weight: high=0.6, low=-0.6, medium=0
        key_w = norm['high'] * 0.6 + norm['low'] * -0.6
        # combine 70% sentiment, 30% keywords
        score = 0.7 * sent_w + 0.3 * key_w
        # thresholds from config
        for lvl, (low, high) in self.stress_thresholds.items():
            if low <= score <= high:
                return lvl
        return 'medium'

class DataPreprocessor:
    """
    Class for preprocessing datasets.
    """
    def __init__(self, config_path):
        self.config_path = config_path
        self._load_config()
        self.data_path = self.config.get('data_path', 'data/')
        self.processed_path = self.config.get('processed_data_path', 'processed_data/')
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.processed_path, exist_ok=True)
        self.text_preprocessor = TextPreprocessor(self.config)

    def _load_config(self):
        try:
            with open(self.config_path) as f:
                self.config = json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self.config = {}

    def process_all_files(self):
        files = [f for f in os.listdir(self.data_path) if f.endswith('.csv')]
        for f in files:
            self.process_file(os.path.join(self.data_path, f))

    def process_file(self, path):
        df = pd.read_csv(path)
        if 'text' not in df:
            logger.error(f"No 'text' column in {path}")
            return
        processed = []
        for _, row in df.iterrows():
            proc = self.text_preprocessor.process_text(row['text'])
            processed.append({**row.to_dict(), **proc})
        pd.DataFrame(processed).to_csv(
            os.path.join(self.processed_path, f"processed_{os.path.basename(path)}"), index=False
        )
        logger.info(f"Saved processed data for {path}")

    def prepare_training_data(self, output_dir='processed_data'):
        os.makedirs(output_dir, exist_ok=True)
        files = [f for f in os.listdir(self.processed_path) if f.startswith('processed_') and f.endswith('.csv')]
        data = []
        for f in files:
            df = pd.read_csv(os.path.join(self.processed_path, f))
            if 'text' in df and 'stress_level' in df:
                data.append(df)
        if not data:
            return None, None
        combined = pd.concat(data).drop_duplicates('text')
        train, val = train_test_split(
            combined, test_size=self.config.get('training', {}).get('validation_split', 0.2),
            random_state=42, stratify=combined['stress_level']
        )
        train.to_csv(os.path.join(output_dir, 'training_data.csv'), index=False)
        val.to_csv(os.path.join(output_dir, 'validation_data.csv'), index=False)
        return train, val
    
