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
from sklearn.utils.class_weight import compute_class_weight

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

        self.max_features = self.config.get('max_features', 10000)
        self.maxlen = self.config.get('maxlen', 200)
        self.embedding_dims = self.config.get('embedding_dims', 300)
        self.model_type = self.config.get('model_type', 'lstm')

        self.tokenizer = None
        self.label_encoder = LabelEncoder()
        self.model = None
        self.history = None

        self.models_dir = self.config.get('models_dir', 'models/')
        os.makedirs(self.models_dir, exist_ok=True)

    def _build_lstm_model(self, input_length: int, num_classes: int) -> Model:
        model = Sequential()
        model.add(Embedding(self.max_features, self.embedding_dims, input_length=input_length))
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Dropout(0.3))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        lr = self.config.get('training', {}).get('learning_rate', 1e-4)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
        logger.info("Built LSTM model architecture")
        return model

    def _build_cnn_lstm_model(self, input_length: int, num_classes: int) -> Model:
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
        lr = self.config.get('training', {}).get('learning_rate', 1e-4)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
        logger.info("Built CNN-LSTM hybrid model architecture")
        return model

    def _build_transformer_model(self, input_length: int, num_classes: int) -> Model:
        from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
        num_heads = 8
        ff_dim = 512

        def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.3):
            x = LayerNormalization(epsilon=1e-6)(inputs)
            x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
            res = x + inputs
            x = LayerNormalization(epsilon=1e-6)(res)
            x = Dense(ff_dim, activation='relu')(x)
            x = Dropout(dropout)(x)
            x = Dense(inputs.shape[-1])(x)
            return x + res

        inputs = Input(shape=(input_length,))
        embedding_layer = Embedding(input_dim=self.max_features, output_dim=self.embedding_dims, input_length=input_length)(inputs)
        x = embedding_layer
        for _ in range(4):
            x = transformer_encoder(x, self.embedding_dims // num_heads, num_heads, ff_dim)
        x = GlobalMaxPooling1D()(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
        lr = self.config.get('training', {}).get('learning_rate', 1e-4)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
        logger.info("Built Transformer model architecture")
        return model

    def preprocess_text(self, texts: List[str], fit_tokenizer: bool = False) -> np.ndarray:
        if self.tokenizer is None or fit_tokenizer:
            self.tokenizer = Tokenizer(num_words=self.max_features, oov_token="<OOV>")
            self.tokenizer.fit_on_texts(texts)
            tokenizer_path = os.path.join(self.models_dir, 'tokenizer.pickle')
            with open(tokenizer_path, 'wb') as handle:
                pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Fitted and saved tokenizer with vocabulary size {len(self.tokenizer.word_index)}")
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.maxlen)
        return padded_sequences

    def encode_labels(self, labels: List[str], fit_encoder: bool = False) -> np.ndarray:
        if fit_encoder:
            # Force consistent label order
            all_labels = ['low', 'medium', 'high']
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(all_labels)
            encoded_labels = self.label_encoder.transform(labels)

            encoder_path = os.path.join(self.models_dir, 'label_encoder.pickle')
            with open(encoder_path, 'wb') as handle:
                pickle.dump(self.label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Fitted and saved label encoder with classes {self.label_encoder.classes_}")
        else:
            encoded_labels = self.label_encoder.transform(labels)
        return encoded_labels


    def build_model(self, num_classes: int) -> None:
        if self.model_type == 'lstm':
            self.model = self._build_lstm_model(self.maxlen, num_classes)
        elif self.model_type == 'cnn_lstm':
            self.model = self._build_cnn_lstm_model(self.maxlen, num_classes)
        elif self.model_type == 'transformer':
            self.model = self._build_transformer_model(self.maxlen, num_classes)
        else:
            logger.warning(f"Unknown model type: {self.model_type}. Using LSTM as default.")
            self.model = self._build_lstm_model(self.maxlen, num_classes)
        self.model.summary(print_fn=logger.info)

    def train(self, texts: List[str], labels: List[str], validation_split: float = None, epochs: int = None, batch_size: int = None) -> Dict[str, List[float]]:
        if validation_split is None:
            validation_split = self.config.get('training', {}).get('validation_split', 0.2)
        if epochs is None:
            epochs = self.config.get('training', {}).get('epochs', 20)
        if batch_size is None:
            batch_size = self.config.get('training', {}).get('batch_size', 32)
        patience = self.config.get('training', {}).get('early_stopping_patience', 3)

        # Rebalance the training data by oversampling the minority class
        from collections import Counter
        from sklearn.utils import resample

        df = pd.DataFrame({'text': texts, 'label': labels})
        count = Counter(labels)
        logger.info(f"Original class distribution: {count}")

        # Upsample all classes to the size of the majority
        max_count = max(count.values())
        upsampled = []
        for label_class in count:
            class_subset = df[df['label'] == label_class]
            class_upsampled = resample(class_subset, replace=True, n_samples=max_count, random_state=42)
            upsampled.append(class_upsampled)

        df_balanced = pd.concat(upsampled).sample(frac=1, random_state=42).reset_index(drop=True)
        texts = df_balanced['text'].tolist()
        labels = df_balanced['label'].tolist()

        logger.info(f"Balanced class distribution: {Counter(labels)}")

        X = self.preprocess_text(texts, fit_tokenizer=True)
        y = self.encode_labels(labels, fit_encoder=True)

        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}
        logger.info(f"Class weights applied: {class_weight_dict}")

        num_classes = len(np.unique(y))
        self.build_model(num_classes)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            ModelCheckpoint(filepath=os.path.join(self.models_dir, 'model_checkpoint.h5'), save_best_only=True, monitor='val_loss'),
            TensorBoard(log_dir=os.path.join(self.models_dir, 'logs'))
        ]

        self.history = self.model.fit(X, y, validation_split=validation_split, epochs=epochs, batch_size=batch_size, callbacks=callbacks, class_weight=class_weight_dict)

        model_path = os.path.join(self.models_dir, 'stress_detection_model.h5')
        self.model.save(model_path)
        logger.info(f"Model trained and saved to {model_path}")
        return self.history.history

    def evaluate(self, texts: List[str], labels: List[str]) -> Dict[str, float]:
        X = self.preprocess_text(texts, fit_tokenizer=False)
        y = self.encode_labels(labels, fit_encoder=False)
        loss, accuracy = self.model.evaluate(X, y)
        y_pred = self.model.predict(X)
        y_pred_classes = np.argmax(y_pred, axis=1)
        report = classification_report(y, y_pred_classes, target_names=self.label_encoder.classes_, output_dict=True)
        cm = confusion_matrix(y, y_pred_classes)
        logger.info(f"Confusion Matrix:\n{cm}")
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
        if isinstance(texts, str):
            texts = [texts]
        X = self.preprocess_text(texts, fit_tokenizer=False)
        predictions = self.model.predict(X)
        results = []
        for i, pred in enumerate(predictions):
            pred_class_idx = np.argmax(pred)
            confidence_score = float(pred[pred_class_idx])
            pred_class = self.label_encoder.classes_[pred_class_idx]
            if confidence_score < 0.6:
                pred_class = "uncertain"
            class_scores = {cls: float(pred[idx]) for idx, cls in enumerate(self.label_encoder.classes_)}
            results.append({
                'text': texts[i],
                'predicted_class': pred_class,
                'confidence': confidence_score,
                'class_scores': class_scores
            })
        return results

    def load_saved_model(self, model_path: str = None, tokenizer_path: str = None, encoder_path: str = None) -> None:
        model_path = model_path or os.path.join(self.models_dir, 'stress_detection_model.h5')
        tokenizer_path = tokenizer_path or os.path.join(self.models_dir, 'tokenizer.pickle')
        encoder_path = encoder_path or os.path.join(self.models_dir, 'label_encoder.pickle')

        if os.path.exists(model_path):
            self.model = load_model(model_path)
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.error(f"Model file not found: {model_path}")

        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'rb') as handle:
                self.tokenizer = pickle.load(handle)
            logger.info(f"Loaded tokenizer from {tokenizer_path}")
        else:
            logger.error(f"Tokenizer file not found: {tokenizer_path}")

        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as handle:
                self.label_encoder = pickle.load(handle)
            logger.info(f"Loaded label encoder from {encoder_path}")
        else:
            logger.error(f"Label encoder file not found: {encoder_path}")
