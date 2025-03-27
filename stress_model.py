import os
import numpy as np
import pandas as pd
import pickle
import json
import logging
from typing import Dict, List, Tuple, Any, Union, Optional
from datetime import datetime

# Machine Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding, Bidirectional
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_development.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StressDetectionModel:
    """
    Neural network model for stress detection in text.
    """
    def __init__(self, config_path: str = None):
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {}
        
        # Set model parameters
        self.max_features = self.config.get('max_features', 10000)  # Size of vocabulary
        self.maxlen = self.config.get('maxlen', 200)               # Max length of sequences
        self.embedding_dims = self.config.get('embedding_dims', 300) # Embedding dimensions
        self.model_type = self.config.get('model_type', 'lstm')     # Model architecture type
        
        # Initialize model components
        self.tokenizer = None
        self.label_encoder = LabelEncoder()
        self.model = None
        self.history = None
        
        # Model saving paths
        self.models_dir = self.config.get('models_dir', 'models/')
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
    def _build_lstm_model(self, input_length: int, num_classes: int) -> Model:
        """
        Build an LSTM-based model for text classification.
        
        Args:
            input_length: Length of input sequences
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        model.add(Embedding(self.max_features, self.embedding_dims, input_length=input_length))
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Dropout(0.3))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        
        # Get learning rate from config or use default
        lr = self.config.get('training', {}).get('learning_rate', 1e-4)
        
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(learning_rate=lr),
            metrics=['accuracy']
        )
        
        logger.info("Built LSTM model architecture")
        return model
    
    def _build_cnn_lstm_model(self, input_length: int, num_classes: int) -> Model:
        """
        Build a CNN-LSTM hybrid model for text classification.
        
        Args:
            input_length: Length of input sequences
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        model.add(Embedding(self.max_features, self.embedding_dims, input_length=input_length))
        model.add(Conv1D(128, 5, activation='relu'))
        model.add(MaxPooling1D(5))
        model.add(Conv1D(128, 5, activation='relu'))
        model.add(MaxPooling1D(5))
        model.add(LSTM(128))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        
        # Get learning rate from config or use default
        lr = self.config.get('training', {}).get('learning_rate', 1e-4)
        
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(learning_rate=lr),
            metrics=['accuracy']
        )
        
        logger.info("Built CNN-LSTM hybrid model architecture")
        return model
    
    def _build_transformer_model(self, input_length: int, num_classes: int) -> Model:
        """
        Build a Transformer-based model for text classification.
        Using TensorFlow's Transformer layers.
        
        Args:
            input_length: Length of input sequences
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        # Import transformer layers
        from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
        
        # Transformer parameters
        num_heads = 8
        ff_dim = 512
        
        # Transformer encoder block
        def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.3):
            # Multi-head self attention
            x = LayerNormalization(epsilon=1e-6)(inputs)
            x = MultiHeadAttention(
                key_dim=head_size, num_heads=num_heads, dropout=dropout
            )(x, x)
            res = x + inputs
            
            # Feed Forward
            x = LayerNormalization(epsilon=1e-6)(res)
            x = Dense(ff_dim, activation='relu')(x)
            x = Dropout(dropout)(x)
            x = Dense(inputs.shape[-1])(x)
            return x + res
        
        # Build model
        inputs = Input(shape=(input_length,))
        embedding_layer = Embedding(
            input_dim=self.max_features,
            output_dim=self.embedding_dims,
            input_length=input_length
        )(inputs)
        
        x = embedding_layer
        
        # Add transformer blocks
        for _ in range(4):
            x = transformer_encoder(x, self.embedding_dims // num_heads, num_heads, ff_dim)
        
        # Global pooling and classification
        x = GlobalMaxPooling1D()(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Get learning rate from config or use default
        lr = self.config.get('training', {}).get('learning_rate', 1e-4)
        
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(learning_rate=lr),
            metrics=['accuracy']
        )
        
        logger.info("Built Transformer model architecture")
        return model
        
    def preprocess_text(self, texts: List[str], fit_tokenizer: bool = False) -> np.ndarray:
        """
        Preprocess text data for the model.
        
        Args:
            texts: List of text strings
            fit_tokenizer: Whether to fit the tokenizer on this data
            
        Returns:
            Padded sequence array
        """
        # Initialize load tokenizer
        if self.tokenizer is None or fit_tokenizer:
            self.tokenizer = Tokenizer(num_words=self.max_features)
            self.tokenizer.fit_on_texts(texts)
            
            # Save tokenizer
            tokenizer_path = os.path.join(self.models_dir, 'tokenizer.pickle')
            with open(tokenizer_path, 'wb') as handle:
                pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
            logger.info(f"Fitted and saved tokenizer with vocabulary size {len(self.tokenizer.word_index)}")
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Pad sequences to same length
        padded_sequences = pad_sequences(sequences, maxlen=self.maxlen)
        
        return padded_sequences
    
    def encode_labels(self, labels: List[str], fit_encoder: bool = False) -> np.ndarray:
        """
        Encode categorical labels to integers.
        
        Args:
            labels: List of label strings
            fit_encoder: Whether to fit the label encoder on this data
            
        Returns:
            Array of encoded labels
        """
        if fit_encoder:
            encoded_labels = self.label_encoder.fit_transform(labels)
            
            # Save label encoder
            encoder_path = os.path.join(self.models_dir, 'label_encoder.pickle')
            with open(encoder_path, 'wb') as handle:
                pickle.dump(self.label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
            logger.info(f"Fitted and saved label encoder with classes {self.label_encoder.classes_}")
        else:
            encoded_labels = self.label_encoder.transform(labels)
            
        return encoded_labels
    
    def build_model(self, num_classes: int) -> None:
        """
        Build the neural network model.
        
        Args:
            num_classes: Number of output classes
        """
        if self.model_type == 'lstm':
            self.model = self._build_lstm_model(self.maxlen, num_classes)
        elif self.model_type == 'cnn_lstm':
            self.model = self._build_cnn_lstm_model(self.maxlen, num_classes)
        elif self.model_type == 'transformer':
            self.model = self._build_transformer_model(self.maxlen, num_classes)
        else:
            logger.warning(f"Unknown model type: {self.model_type}. Using LSTM as default.")
            self.model = self._build_lstm_model(self.maxlen, num_classes)
        
        # Print model summary
        self.model.summary(print_fn=logger.info)
    
    def train(self, 
              texts: List[str], 
              labels: List[str], 
              validation_split: float = None,
              epochs: int = None,
              batch_size: int = None) -> Dict[str, List[float]]:
        """
        Train the model on the provided data.
        
        Args:
            texts: List of text strings
            labels: List of label strings
            validation_split: Proportion of data to use for validation
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Training history
        """
        # Get training parameters from config if not specified
        if validation_split is None:
            validation_split = self.config.get('training', {}).get('validation_split', 0.2)
        if epochs is None:
            epochs = self.config.get('training', {}).get('epochs', 20)
        if batch_size is None:
            batch_size = self.config.get('training', {}).get('batch_size', 32)
            
        # Get early stopping patience from config
        patience = self.config.get('training', {}).get('early_stopping_patience', 3)
        
        # Preprocess data
        X = self.preprocess_text(texts, fit_tokenizer=True)
        y = self.encode_labels(labels, fit_encoder=True)
        
        # Build model
        num_classes = len(set(labels))
        self.build_model(num_classes)
        
        # Prepare callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            ModelCheckpoint(
                filepath=os.path.join(self.models_dir, 'model_checkpoint.h5'),
                save_best_only=True,
                monitor='val_loss'
            ),
            TensorBoard(log_dir=os.path.join(self.models_dir, 'logs'))
        ]
        
        # Train model
        self.history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        # Save final model
        model_path = os.path.join(self.models_dir, 'stress_detection_model.h5')
        self.model.save(model_path)
        logger.info(f"Model trained and saved to {model_path}")
        
        return self.history.history
    
    def evaluate(self, texts: List[str], labels: List[str]) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            texts: List of text strings
            labels: List of label strings
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Preprocess data
        X = self.preprocess_text(texts, fit_tokenizer=False)
        y = self.encode_labels(labels, fit_encoder=False)
        
        # Evaluate model
        loss, accuracy = self.model.evaluate(X, y)
        
        # Get predictions
        y_pred = self.model.predict(X)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Generate classification report
        report = classification_report(y, y_pred_classes, target_names=self.label_encoder.classes_, output_dict=True)
        
        # Log confusion matrix
        cm = confusion_matrix(y, y_pred_classes)
        logger.info(f"Confusion Matrix:\n{cm}")
        
        # Combine metrics
        metrics = {
            'loss': loss,
            'accuracy': accuracy,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score'],
            'class_report': report
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def predict(self, texts: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Make predictions on new text data.
        
        Args:
            texts: A text string or list of text strings
            
        Returns:
            List of dictionaries with prediction results
        """
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
            
        # Preprocess input texts
        X = self.preprocess_text(texts, fit_tokenizer=False)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Process prediction results
        results = []
        for i, pred in enumerate(predictions):
            # Get predicted class
            pred_class_idx = np.argmax(pred)
            pred_class = self.label_encoder.classes_[pred_class_idx]
            
            # Get confidence scores for all classes
            class_scores = {cls: float(pred[idx]) for idx, cls in enumerate(self.label_encoder.classes_)}
            
            # Create result dictionary
            result = {
                'text': texts[i],
                'predicted_class': pred_class,
                'confidence': float(pred[pred_class_idx]),
                'class_scores': class_scores
            }
            
            results.append(result)
            
        return results
    
    def load_saved_model(self, model_path: str = None, tokenizer_path: str = None, encoder_path: str = None) -> None:
        """
        Load a saved model, tokenizer, and label encoder.
        
        Args:
            model_path: Path to the saved model
            tokenizer_path: Path to the saved tokenizer
            encoder_path: Path to the saved label encoder
        """
        # Set default paths if not provided
        if model_path is None:
            model_path = os.path.join(self.models_dir, 'stress_detection_model.h5')
        if tokenizer_path is None:
            tokenizer_path = os.path.join(self.models_dir, 'tokenizer.pickle')
        if encoder_path is None:
            encoder_path = os.path.join(self.models_dir, 'label_encoder.pickle')
        
        # Load model
        if os.path.exists(model_path):
            self.model = load_model(model_path)
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.error(f"Model file not found: {model_path}")
            
        # Load tokenizer
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'rb') as handle:
                self.tokenizer = pickle.load(handle)
            logger.info(f"Loaded tokenizer from {tokenizer_path}")
        else:
            logger.error(f"Tokenizer file not found: {tokenizer_path}")
            
        # Load label encoder
        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as handle:
                self.label_encoder = pickle.load(handle)
            logger.info(f"Loaded label encoder from {encoder_path}")
        else:
            logger.error(f"Label encoder file not found: {encoder_path}")


# Example 
if __name__ == "__main__":
    # Example of training a stress detection model
    detector = StressDetectionModel("config.json")
    
    # Sample data (in a real scenario, this would be loaded from preprocessed files)
    texts = [
        "I'm feeling overwhelmed with all these deadlines.",
        "Just had a great day at the beach, feeling relaxed.",
        "This project is stressing me out, can't sleep well.",
        "Had a wonderful time with friends today.",
        "The pressure at work is unbearable right now."
    ]
    labels = ["high", "low", "high", "low", "high"]
    
    # Train model on sample data
    detector.train(texts, labels)
    
    # Make prediction
    prediction = detector.predict("I'm worried about the upcoming presentation")
    print(prediction)