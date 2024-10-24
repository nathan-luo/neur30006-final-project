import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np
from typing import Tuple, Dict, Any, List
import json
import os
from datetime import datetime

class ModelManager:
    def __init__(self, input_shape: int = 784, num_classes: int = 10):
        """
        Initialize ModelManager with basic architecture parameters.
        
        Args:
            input_shape: Input dimension (default 784 for MNIST)
            num_classes: Number of output classes (default 10 for MNIST)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.model_config = None

    def build_model(self, 
                   hidden_layers: List[int] = [128, 64],
                   dropout_rates: List[float] = [0.3, 0.3],
                   use_batch_norm: bool = True,
                   learning_rate: float = 0.001) -> None:
        """
        Build the neural network model.
        
        Args:
            hidden_layers: List of neurons in each hidden layer
            dropout_rates: List of dropout rates for each hidden layer
            use_batch_norm: Whether to use batch normalization
            learning_rate: Learning rate for Adam optimizer
        """
        model = Sequential()
        
        # Input layer
        model.add(Dense(hidden_layers[0], input_shape=(self.input_shape,), 
                       activation='relu'))
        if use_batch_norm:
            model.add(BatchNormalization())
        if dropout_rates[0] > 0:
            model.add(Dropout(dropout_rates[0]))
            
        # Hidden layers
        for units, dropout_rate in zip(hidden_layers[1:], dropout_rates[1:]):
            model.add(Dense(units, activation='relu'))
            if use_batch_norm:
                model.add(BatchNormalization())
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(self.num_classes, activation='softmax'))
        
        # Compile model
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        self.model = model
        self.model_config = {
            'hidden_layers': hidden_layers,
            'dropout_rates': dropout_rates,
            'use_batch_norm': use_batch_norm,
            'learning_rate': learning_rate
        }

    def get_callbacks(self, 
                     patience: int = 10,
                     model_path: str = None) -> List[tf.keras.callbacks.Callback]:
        """
        Get training callbacks.
        
        Args:
            patience: Patience for early stopping
            model_path: Path to save best model
        """
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
        ]
        
        if model_path:
            callbacks.append(
                ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)
            )
        
        return callbacks

    def train(self, 
             X_train: np.ndarray,
             y_train: np.ndarray,
             validation_data: Tuple[np.ndarray, np.ndarray] = None,
             batch_size: int = 128,
             epochs: int = 100,
             callbacks: List[tf.keras.callbacks.Callback] = None) -> Dict:
        """
        Train the model with given data.
        
        Args:
            X_train: Training data
            y_train: Training labels
            validation_data: Tuple of (X_val, y_val)
            batch_size: Batch size for training
            epochs: Maximum number of epochs
            callbacks: List of callbacks
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
            
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history.history

    def train_with_augmentation(self,
                              X_train: np.ndarray,
                              y_train: np.ndarray,
                              augment_func: callable,
                              validation_data: Tuple[np.ndarray, np.ndarray] = None,
                              batch_size: int = 128,
                              epochs: int = 100,
                              callbacks: List[tf.keras.callbacks.Callback] = None) -> Dict:
        """
        Train the model with real-time data augmentation.
        
        Args:
            X_train: Training data
            y_train: Training labels
            augment_func: Function that generates augmented data
            validation_data: Tuple of (X_val, y_val)
            batch_size: Batch size for training
            epochs: Maximum number of epochs
            callbacks: List of callbacks
        """
        def data_generator():
            while True:
                indices = np.random.randint(0, len(X_train), batch_size)
                batch_x = X_train[indices]
                batch_y = y_train[indices]
                
                # Apply augmentation
                augmented_x = augment_func(batch_x)
                yield augmented_x, batch_y

        steps_per_epoch = len(X_train) // batch_size
        
        self.history = self.model.fit(
            data_generator(),
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history.history

    def evaluate(self, 
                X_test: np.ndarray,
                y_test: np.ndarray,
                verbose: int = 1) -> Tuple[float, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test data
            y_test: Test labels
            verbose: Verbosity level
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
            
        return self.model.evaluate(X_test, y_test, verbose=verbose)

    def predict(self, 
               X: np.ndarray,
               batch_size: int = None) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input data
            batch_size: Batch size for prediction
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
            
        return self.model.predict(X, batch_size=batch_size)

    def save(self, 
            base_path: str = 'models',
            include_weights: bool = True) -> str:
        """
        Save the model and its configuration.
        
        Args:
            base_path: Base directory to save model
            include_weights: Whether to save model weights
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(base_path, f'model_{timestamp}')
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model configuration
        config_path = os.path.join(model_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.model_config, f)
        
        # Save training history if exists
        if self.history is not None:
            history_path = os.path.join(model_dir, 'history.json')
            with open(history_path, 'w') as f:
                json.dump(self.history.history, f)
        
        # Save model weights if requested
        if include_weights:
            weights_path = os.path.join(model_dir, 'model.h5')
            save_model(self.model, weights_path)
        
        return model_dir

    @classmethod
    def load(cls, 
            model_dir: str) -> 'ModelManager':
        """
        Load a saved model and its configuration.
        
        Args:
            model_dir: Directory containing saved model
        """
        # Load configuration
        config_path = os.path.join(model_dir, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create new instance
        instance = cls()
        
        # Build model with loaded configuration
        instance.build_model(
            hidden_layers=config['hidden_layers'],
            dropout_rates=config['dropout_rates'],
            use_batch_norm=config['use_batch_norm'],
            learning_rate=config['learning_rate']
        )
        
        # Load weights if they exist
        weights_path = os.path.join(model_dir, 'model.h5')
        if os.path.exists(weights_path):
            instance.model = load_model(weights_path)
        
        # Load history if it exists
        history_path = os.path.join(model_dir, 'history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                instance.history = type('History', (), {'history': json.load(f)})
        
        return instance

    def get_model_summary(self) -> str:
        """Get model summary as string."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
            
        string_list = []
        self.model.summary(print_fn=lambda x: string_list.append(x))
        return '\n'.join(string_list)