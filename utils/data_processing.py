import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

def load_mnist_from_csv():
    """Load and preprocess MNIST data from CSV files"""
    # Load CSV files
    train_df = pd.read_csv('data/mnist_train.csv', header=None)
    test_df = pd.read_csv('data/mnist_test.csv', header=None)
    
    # Separate labels and features
    y_train = train_df[0]
    X_train = train_df.drop(0, axis=1)
    
    y_test = test_df[0]
    X_test = test_df.drop(0, axis=1)
    
    # Convert to numpy arrays and normalize pixel values
    X_train = X_train.values.astype('float32') / 255.0
    X_test = X_test.values.astype('float32') / 255.0
    
    # One-hot encode labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return X_train, y_train, X_test, y_test


def visualize_digit(image, label=None):
    """Visualize a single MNIST digit"""
    # Reshape if needed
    if image.shape == (784,):
        image = image.reshape(28, 28)
    
    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    if label is not None:
        plt.title(f"Label: {np.argmax(label)}")
    plt.show()


def visualize_multiple(images, labels=None, num_images=10):
    """Visualize multiple digits in a row"""
    plt.figure(figsize=(2*num_images, 2))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        img = images[i].reshape(28, 28)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        if labels is not None:
            plt.title(str(np.argmax(labels[i])))
    plt.tight_layout()
    plt.show()