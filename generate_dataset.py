import os
import json
import pandas as pd
import numpy as np
import logging
import random
from datetime import datetime

logger = logging.getLogger(__name__)

class DatasetGenerator:
    """
    Generate synthetic datasets for training and testing stress detection models.
    """
    
    def __init__(self, config_path):
        """
        Initialize the dataset generator.
        
        Args:
            config_path (str): Path to configuration file.
        """
        self.config_path = config_path
        self.load_config()
        
        # Create directories if they don't exist
        os.makedirs(self.config.get('data_path', 'data/'), exist_ok=True)
        
        # Define templates for different stress levels
        self.templates = {
            'low': [
                "I'm feeling {positive_adj} today. {positive_phrase}.",
                "Had a {positive_adj} day at {location}. {positive_phrase}.",
                "Just finished {positive_activity} and feeling {positive_adj}.",
                "Everything seems {positive_adj} right now. {positive_phrase}.",
                "I've been {positive_verb} all day and it's been {positive_adj}.",
                "My {time_period} has been going {positive_adv}. {positive_phrase}.",
                "{positive_phrase}. I feel {positive_adj} about everything.",
                "Taking time to {positive_activity} has made me feel {positive_adj}.",
                "I'm in a {positive_adj} mood today. {positive_phrase}.",
                "Life is {positive_adj} right now. {positive_phrase}."
            ],
            'medium': [
                "Feeling a bit {negative_adj} about {stress_source}, but managing.",
                "Had some {negative_adj} moments today at {location}, but overall okay.",
                "{stress_source} is causing some {negative_adj} feelings, but I'm coping.",
                "A bit {negative_adj} about {stress_source}, but trying to stay positive.",
                "Things are somewhat {negative_adj} with {stress_source}, but not too bad.",
                "My {time_period} has been {negative_adj}, but I'm handling it.",
                "I'm dealing with some {negative_adj} issues with {stress_source}.",
                "Experiencing some {negative_noun} with {stress_source}, but managing.",
                "I'm a little {negative_adj} about {stress_source}, but it's temporary.",
                "Some {negative_adj} situations at {location}, but I'll get through it."
            ],
            'high': [
                "I'm extremely {negative_adj} about {stress_source}. {negative_phrase}.",
                "The {negative_noun} from {stress_source} is unbearable. {negative_phrase}.",
                "I can't handle the {negative_noun} from {stress_source} anymore. {negative_phrase}.",
                "Everything about {stress_source} is making me feel {negative_adj}. {negative_phrase}.",
                "I'm completely {negative_adj} by {stress_source}. {negative_phrase}.",
                "The {negative_noun} is overwhelming me. {negative_phrase}.",
                "I don't know how to cope with this {negative_noun}. {negative_phrase}.",
                "I'm at my breaking point with {stress_source}. {negative_phrase}.",
                "This {negative_noun} is destroying me. {negative_phrase}.",
                "I feel absolutely {negative_adj} about everything related to {stress_source}."
            ]
        }
        
        # Define word lists for template filling
        self.word_lists = {
            'positive_adj': [
                'relaxed', 'peaceful', 'calm', 'content', 'happy', 'great', 'wonderful',
                'excellent', 'fantastic', 'balanced', 'refreshed', 'energized', 'optimistic',
                'positive', 'good', 'joyful', 'satisfied', 'tranquil', 'serene', 'pleasant'
            ],
            'positive_phrase': [
                'Everything is going smoothly', 'No complaints at all', 'Life is good',
                'Feeling in control of things', 'Nothing to worry about', 'Enjoying the moment',
                'Taking everything in stride', 'Appreciating the little things',
                'Feeling grateful for what I have', 'Things are falling into place',
                'Finding joy in simple pleasures', 'Maintaining a good perspective',
                'Focusing on the positive', 'Keeping a balanced outlook',
                'Enjoying the peace and quiet'
            ],
            'positive_activity': [
                'meditation', 'yoga', 'a relaxing walk', 'deep breathing exercises',
                'a good workout', 'spending time with friends', 'reading a good book',
                'listening to music', 'cooking a nice meal', 'gardening',
                'enjoying nature', 'practicing mindfulness', 'taking a hot bath',
                'watching a good movie', 'playing with my pet', 'painting', 'journaling'
            ],
            'positive_verb': [
                'relaxing', 'smiling', 'enjoying myself', 'laughing', 'feeling content',
                'appreciating life', 'staying positive', 'feeling grateful', 'feeling balanced',
                'taking it easy', 'going with the flow', 'embracing the moment'
            ],
            'positive_adv': [
                'smoothly', 'wonderfully', 'excellently', 'perfectly', 'fantastically',
                'better than expected', 'incredibly well', 'remarkably well',
                'surprisingly well', 'exceptionally well'
            ],
            'negative_adj': [
                'stressed', 'anxious', 'worried', 'overwhelmed', 'frustrated', 'exhausted',
                'nervous', 'tense', 'uneasy', 'concerned', 'troubled', 'distressed',
                'drained', 'pressured', 'agitated', 'irritated', 'upset', 'restless',
                'apprehensive', 'fearful'
            ],
            'negative_phrase': [
                "I don't know how much more I can take", "It's affecting my sleep",
                "I can't focus on anything else", "My mind won't stop racing",
                "I feel like I'm losing control", "Everything feels overwhelming",
                "I'm constantly on edge", "I can't see an end to this",
                "It's taking a toll on my health", "I'm at my breaking point",
                "I feel like I'm drowning", "I can't escape this feeling",
                "My anxiety is through the roof", "I'm completely burnt out",
                "I feel hopeless about the situation"
            ],
            'negative_noun': [
                'stress', 'anxiety', 'pressure', 'tension', 'burden', 'worry',
                'exhaustion', 'strain', 'frustration', 'overwhelm', 'dread',
                'nervousness', 'apprehension', 'unease', 'distress'
            ],
            'stress_source': [
                'work', 'my job', 'school', 'exams', 'deadlines', 'finances',
                'relationships', 'family issues', 'health concerns', 'the future',
                'my boss', 'my colleagues', 'my studies', 'my responsibilities',
                'my living situation', 'my commute', 'current events', 'the news',
                'social media', 'personal projects'
            ],
            'location': [
                'work', 'school', 'home', 'the office', 'university', 'the gym',
                'the store', 'my apartment', 'my parent\'s house', 'the conference',
                'the meeting', 'the doctor\'s office', 'the party', 'the event'
            ],
            'time_period': [
                'day', 'week', 'month', 'morning', 'afternoon', 'evening',
                'weekend', 'workday', 'vacation', 'semester', 'year', 'life'
            ]
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
    
    def generate_text(self, stress_level):
        """
        Generate a synthetic text for a given stress level.
        
        Args:
            stress_level (str): Stress level ('low', 'medium', or 'high').
            
        Returns:
            str: Generated text.
        """
        # Select template
        template = random.choice(self.templates[stress_level])
        
        # Fill template with random words
        for key, word_list in self.word_lists.items():
            if '{' + key + '}' in template:
                template = template.replace('{' + key + '}', random.choice(word_list))
        
        return template
    
    def generate_dataset(self, size, distribution=None):
        """
        Generate a synthetic dataset.
        
        Args:
            size (int): Number of samples to generate.
            distribution (dict, optional): Distribution of stress levels.
                                        Defaults to {'low': 0.33, 'medium': 0.34, 'high': 0.33}.
            
        Returns:
            pd.DataFrame: Generated dataset.
        """
        if distribution is None:
            distribution = {'low': 0.33, 'medium': 0.34, 'high': 0.33}
        
        # Calculate number of samples for each level
        level_counts = {
            level: int(size * prob) 
            for level, prob in distribution.items()
        }
        
        # Adjust for rounding errors
        total = sum(level_counts.values())
        if total < size:
            level_counts['medium'] += size - total
        
        data = []
        
        # Generate samples for each level
        for level, count in level_counts.items():
            for _ in range(count):
                text = self.generate_text(level)
                data.append({
                    'text': text,
                    'stress_level': level
                })
        
        # Shuffle data
        random.shuffle(data)
        
        return pd.DataFrame(data)
    
    def generate_realistic_dataset(self, size):
        """
        Generate a more realistic dataset with contexts and timestamps.
        
        Args:
            size (int): Number of samples to generate.
            
        Returns:
            pd.DataFrame: Generated dataset.
        """
        # Default to even distribution
        distribution = {'low': 0.33, 'medium': 0.34, 'high': 0.33}
        
        # Generate base dataset
        df = self.generate_dataset(size, distribution)
        
        # Add additional realistic features
        timestamps = []
        sources = []
        authors = []
        contexts = []
        locations = []
        
        # Create fake author names
        first_names = [
            'James', 'Mary', 'John', 'Patricia', 'Robert', 'Jennifer', 'Michael', 'Linda',
            'William', 'Elizabeth', 'David', 'Susan', 'Richard', 'Jessica', 'Joseph', 'Sarah',
            'Thomas', 'Karen', 'Charles', 'Nancy', 'Daniel', 'Lisa', 'Matthew', 'Margaret'
        ]
        last_names = [
            'Smith', 'Johnson', 'Williams', 'Jones', 'Brown', 'Davis', 'Miller', 'Wilson',
            'Moore', 'Taylor', 'Anderson', 'Thomas', 'Jackson', 'White', 'Harris', 'Martin',
            'Thompson', 'Garcia', 'Martinez', 'Robinson', 'Clark', 'Rodriguez', 'Lewis', 'Lee'
        ]
        
        # Create sources
        possible_sources = ['text_message', 'email', 'survey', 'social_media', 'chat', 'forum', 'blog']
        
        # Create contexts
        possible_contexts = ['work', 'personal', 'academic', 'health', 'relationship', 'family', 'financial']
        
        # Create locations
        possible_locations = ['home', 'office', 'school', 'transit', 'vacation', 'restaurant', 'gym']
        
        # Generate random features
        for _ in range(len(df)):
            # Random timestamp within the last year
            days_ago = random.randint(0, 365)
            timestamp = datetime.now() - pd.Timedelta(days=days_ago)
            timestamps.append(timestamp)
            
            # Random source
            sources.append(random.choice(possible_sources))
            
            # Random author
            author = f"{random.choice(first_names)} {random.choice(last_names)}"
            authors.append(author)
            
            # Random context
            contexts.append(random.choice(possible_contexts))
            
            # Random location
            locations.append(random.choice(possible_locations))
        
        # Add to dataframe
        df['timestamp'] = timestamps
        df['source'] = sources
        df['author'] = authors
        df['context'] = contexts
        df['location'] = locations
        
        return df
    
    def generate_and_save_datasets(self, train_size=800, val_size=100, test_size=100):
        """
        Generate and save training, validation, and test datasets.
        
        Args:
            train_size (int, optional): Size of training dataset. Defaults to 800.
            val_size (int, optional): Size of validation dataset. Defaults to 100.
            test_size (int, optional): Size of test dataset. Defaults to 100.
            
        Returns:
            tuple: Paths to saved datasets.
        """
        # Generate datasets
        train_df = self.generate_dataset(train_size)
        val_df = self.generate_dataset(val_size)
        test_df = self.generate_dataset(test_size)
        
        # Save datasets
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        train_path = os.path.join(
            self.config.get('data_path', 'data/'),
            f'synthetic_train_{timestamp}.csv'
        )
        val_path = os.path.join(
            self.config.get('data_path', 'data/'),
            f'synthetic_val_{timestamp}.csv'
        )
        test_path = os.path.join(
            self.config.get('data_path', 'data/'),
            f'synthetic_test_{timestamp}.csv'
        )
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"Training data ({len(train_df)} samples) saved to {train_path}")
        logger.info(f"Validation data ({len(val_df)} samples) saved to {val_path}")
        logger.info(f"Test data ({len(test_df)} samples) saved to {test_path}")
        
        return train_path, val_path, test_path
    
    def generate_and_save_realistic_dataset(self, size=500):
        """
        Generate and save a realistic dataset with additional features.
        
        Args:
            size (int, optional): Size of dataset. Defaults to 500.
            
        Returns:
            str: Path to saved dataset.
        """
        # Generate dataset
        df = self.generate_realistic_dataset(size)
        
        # Save dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_path = os.path.join(
            self.config.get('data_path', 'data/'),
            f'realistic_data_{timestamp}.csv'
        )
        
        df.to_csv(output_path, index=False)
        
        logger.info(f"Realistic data ({len(df)} samples) saved to {output_path}")
        
        return output_path


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test the dataset generator
    if os.path.exists("config.json"):
        generator = DatasetGenerator("config.json")
        
        # Generate and save datasets
        train_path, val_path, test_path = generator.generate_and_save_datasets(
            train_size=10,
            val_size=5,
            test_size=5
        )
        
        # Generate and save realistic dataset
        realistic_path = generator.generate_and_save_realistic_dataset(size=10)
        
        # Display sample data
        try:
            test_df = pd.read_csv(test_path)
            print("\nSample test data:")
            for i, row in test_df.head(3).iterrows():
                print(f"\nText: {row['text']}")
                print(f"Stress Level: {row['stress_level']}")
            
            realistic_df = pd.read_csv(realistic_path)
            print("\nSample realistic data:")
            for i, row in realistic_df.head(3).iterrows():
                print(f"\nText: {row['text']}")
                print(f"Stress Level: {row['stress_level']}")
                print(f"Context: {row['context']}")
                print(f"Source: {row['source']}")
        except Exception as e:
            logger.error(f"Error displaying sample data: {str(e)}")
    else:
        logger.error("Config file not found")