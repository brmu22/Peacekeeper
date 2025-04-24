import os
import argparse
import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from data_collection import DataCollectionOrchestrator
from generate_dataset import DatasetGenerator
from text_preprocessing import DataPreprocessor, TextPreprocessor
from stress_model import StressDetectionModel
from sklearn.utils import resample

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("peacekeeper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def balance_dataset(df):
    """
    Upsample classes to balance the dataset.
    """
    # Separate by stress level
    low = df[df['stress_level'] == 'low']
    medium = df[df['stress_level'] == 'medium']
    high = df[df['stress_level'] == 'high']

    max_size = max(len(low), len(medium), len(high))

    low_upsampled = resample(low, replace=True, n_samples=max_size, random_state=42)
    medium_upsampled = resample(medium, replace=True, n_samples=max_size, random_state=42)
    high_upsampled = resample(high, replace=True, n_samples=max_size, random_state=42)

    # Combine and shuffle
    balanced_df = pd.concat([low_upsampled, medium_upsampled, high_upsampled])
    return balanced_df.sample(frac=1).reset_index(drop=True)

def setup_directories():
    """
    Create necessary directories if they don't exist.
    """
    dirs = ['data', 'processed_data', 'models', 'results', 'logs']
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            logger.info(f"Created directory: {dir_name}")

def collect_data(config_path):
    """
    Collect data for the model.
    """
    logger.info("Starting data collection")
    orchestrator = DataCollectionOrchestrator(config_path)
    orchestrator.collect_all_data()
    logger.info("Data collection completed")

def generate_synthetic_data(config_path):
    """
    Generate synthetic data for the model.
    """
    logger.info("Generating synthetic data")
    generator = DatasetGenerator(config_path)
    train_path, val_path, test_path = generator.generate_and_save_datasets(
        train_size=800, 
        val_size=100, 
        test_size=100
    )
    realistic_path = generator.generate_and_save_realistic_dataset(size=500)
    logger.info(f"Generated synthetic data at {train_path}, {val_path}, {test_path}, {realistic_path}")
    return train_path, val_path, test_path

def preprocess_data(config_path):
    """
    Preprocess data for model training.
    """
    logger.info("Starting data preprocessing")
    preprocessor = DataPreprocessor(config_path)
    preprocessor.process_all_files()
    train_df, val_df = preprocessor.prepare_training_data('models')
    logger.info(f"Preprocessing completed. Training data: {len(train_df)} samples, Validation data: {len(val_df)} samples")
    return train_df, val_df

def train_model(config_path, train_df=None, val_df=None):
    """
    Train the stress detection model.
    """
    logger.info("Starting model training")
    model = StressDetectionModel(config_path)
    
    if train_df is None or val_df is None:
        # Load data if not provided
        logger.info("Loading data for training")
        try:
            train_df = pd.read_csv('models/training_data.csv')
            val_df = pd.read_csv('models/validation_data.csv')
        except FileNotFoundError:
            logger.error("Training or validation data not found. Please run preprocess_data first.")
            return
    
    # Balance the training data
    train_df = balance_dataset(train_df)

    # Fit label encoder BEFORE validation to avoid unseen label errors
    model.label_encoder.fit(train_df['stress_level'])

    print("Label classes:", model.label_encoder.classes_)
    print("Number of classes:", len(model.label_encoder.classes_))


    # Extract text and labels
    train_texts = train_df['text'].tolist()
    train_labels = train_df['stress_level'].tolist()
    
    val_texts = val_df['text'].tolist()
    val_labels = val_df['stress_level'].tolist()
    
    # Train model
    history = model.train(train_texts, train_labels)
    
    # Evaluate model
    metrics = model.evaluate(val_texts, val_labels)
    
    logger.info(f"Model training completed. Validation accuracy: {metrics['accuracy']:.4f}")
    
    # Plot training history
    plot_training_history(history)
    
    # Return model and metrics
    return model, metrics

def plot_training_history(history):
    """
    Plot training history.
    """
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'results/training_history_{timestamp}.png'
    plt.savefig(filename)
    logger.info(f"Training history plot saved to {filename}")
    
    plt.close()

def predict_stress(config_path, text):
    """
    Predict stress level for a given text.
    """
    model = StressDetectionModel(config_path)

    # Try loading the model
    model.load_saved_model()

    # If model still None, train it
    if model.model is None:
        logger.warning("Model not loaded. Training a new model...")
        from generate_dataset import DatasetGenerator
        from text_preprocessing import DataPreprocessor
        from main import train_model

        # Generate synthetic data
        generator = DatasetGenerator(config_path)
        generator.generate_and_save_datasets(train_size=200, val_size=50, test_size=50)

        # Preprocess data
        preprocessor = DataPreprocessor(config_path)
        train_df, val_df = preprocessor.prepare_training_data('models')

        # Train model
        trained_model, _ = train_model(config_path, train_df, val_df)
        model = trained_model if trained_model else model

        # Reload everything to ensure predict will work
        model = StressDetectionModel(config_path)
        model.load_saved_model()

    # Basic text processing
    text_processor = TextPreprocessor()
    basic_result = text_processor.process_text(text)

    # Predict
    prediction = model.predict(text)

    return {
        'text': text,
        'basic_analysis': basic_result,
        'model_prediction': prediction[0] if prediction else None
    }
    
def analyze_sample_texts(config_path):
    """
    Analyze some sample texts to demonstrate the model.
    """
    sample_texts = [
        "I'm feeling great today. Everything is going smoothly and I'm enjoying my work.",
        "It's been a busy day but I'm managing fine. Some minor concerns but nothing serious.",
        "I'm extremely stressed about this project deadline. The pressure is overwhelming and I can't sleep well."
    ]
    
    results = []
    for text in sample_texts:
        result = predict_stress(config_path, text)
        results.append(result)
        
        print(f"\nText: {text}")
        print(f"Basic Analysis - Stress Level: {result['basic_analysis']['stress_level']}")
        print(f"Basic Analysis - Sentiment: {result['basic_analysis']['vader_scores']['compound']:.4f}")
        
        if result['model_prediction']:
            print(f"Model Prediction: {result['model_prediction']['predicted_class']} (Confidence: {result['model_prediction']['confidence']:.4f})")
        
    return results

def main():
    """
    Main function to run the PeaceKeeper system.
    """
    parser = argparse.ArgumentParser(description='PeaceKeeper - Stress Detection System')
    parser.add_argument('--config', default='config.json', help='Path to configuration file')
    parser.add_argument('--collect', action='store_true', help='Collect data')
    parser.add_argument('--generate', action='store_true', help='Generate synthetic data')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess data')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--analyze', action='store_true', help='Analyze sample texts')
    parser.add_argument('--text', type=str, help='Text to analyze')
    
    args = parser.parse_args()
    
    # Create directories
    setup_directories()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        return
    
    # Perform requested actions
    if args.collect:
        collect_data(args.config)
    
    if args.generate:
        generate_synthetic_data(args.config)
    
    if args.preprocess:
        preprocess_data(args.config)
    
    if args.train:
        train_model(args.config)
    
    if args.analyze:
        analyze_sample_texts(args.config)
    
    if args.text:
        result = predict_stress(args.config, args.text)
        print(f"\nText: {result['text']}")
        print(f"Basic Analysis - Stress Level: {result['basic_analysis']['stress_level']}")
        print(f"Basic Analysis - Sentiment: {result['basic_analysis']['vader_scores']['compound']:.4f}")
        
        if result['model_prediction']:
            print(f"Model Prediction: {result['model_prediction']['predicted_class']} (Confidence: {result['model_prediction']['confidence']:.4f})")

if __name__ == "__main__":
    main()