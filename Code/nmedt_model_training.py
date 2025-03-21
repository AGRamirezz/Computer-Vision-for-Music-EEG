#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NMEDT Model Training

This script trains a CNN model for EEG classification using preprocessed data.
The model classifies EEG data into 10 different classes (subjects).

Usage:
    - Ensure data files (x_train.npy, x_test.npy, y_train2.npy, y_test2.npy) are in the specified path
    - Run the script to train the model and evaluate performance
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, roc_auc_score, confusion_matrix
)
import gc

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_data(data_path):
    """
    Load training and testing data from specified path.
    
    Args:
        data_path (str): Path to the directory containing data files
        
    Returns:
        tuple: X_train, X_test, y_train, y_test arrays
    """
    X_train = np.load(os.path.join(data_path, "x_train.npy"))
    X_test = np.load(os.path.join(data_path, "x_test.npy"))
    y_train = np.load(os.path.join(data_path, "y_train2.npy"))
    y_test = np.load(os.path.join(data_path, "y_test2.npy"))
    
    print(f"X shapes: {X_train.shape}, {X_test.shape}")
    print(f"y shapes: {y_train.shape}, {y_test.shape}")
    
    return X_train, X_test, y_train, y_test

def build_model(input_shape=(125, 125, 1), batch_size=500):
    """
    Build and compile the CNN model for EEG classification.
    
    Args:
        input_shape (tuple): Shape of input data (height, width, channels)
        batch_size (int): Batch size for model input
        
    Returns:
        keras.Model: Compiled model ready for training
    """
    kernel_init = keras.initializers.he_uniform(seed=1369)
    kernel_reg = keras.regularizers.l2(0.000114)
    
    inputs = keras.Input(shape=input_shape, batch_size=batch_size)
    
    # First convolutional block
    x = layers.Conv2D(
        kernel_size=(4, 4), strides=(2, 2), filters=32, activation="relu",
        kernel_initializer=kernel_init, kernel_regularizer=kernel_reg, 
        padding="same", name="layer1"
    )(inputs)
    x = layers.BatchNormalization()(x)
    
    # Second convolutional block
    x = layers.Conv2D(
        kernel_size=(4, 4), strides=(2, 2), filters=64, activation="relu",
        kernel_initializer=kernel_init, kernel_regularizer=kernel_reg, 
        padding="same", name="layer2"
    )(x)
    x = layers.BatchNormalization()(x)
    
    # Third convolutional block
    x = layers.Conv2D(
        kernel_size=(4, 4), strides=(2, 2), filters=128, activation="relu",
        kernel_initializer=kernel_init, kernel_regularizer=kernel_reg, 
        padding="same", name="layer3"
    )(x)
    x = layers.BatchNormalization()(x)
    
    # Global pooling and dense layers
    x = layers.GlobalAveragePooling2D(data_format='channels_last')(x)
    x = layers.Dense(100, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(10, activation="softmax")(x)
    
    model = keras.Model(inputs, x, name="eeg_classifier")
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=0.009)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['categorical_accuracy']
    )
    
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=200, batch_size=500):
    """
    Train the model and return training history.
    
    Args:
        model (keras.Model): Compiled model
        X_train, y_train: Training data and labels
        X_test, y_test: Testing data and labels
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        
    Returns:
        history: Training history object
    """
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test)
    )
    return history

def plot_training_history(history):
    """
    Plot training and validation loss.
    
    Args:
        history: Training history object from model.fit()
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig('model_loss.png')
    plt.show()

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance using various metrics.
    
    Args:
        model (keras.Model): Trained model
        X_test: Test data
        y_test: Test labels (one-hot encoded)
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Get predictions
    y_pred_proba = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_proba, axis=-1)
    
    # Convert one-hot encoded test labels to class indices
    y_test_classes = np.argmax(y_test, axis=-1)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test_classes, y_pred_classes),
        'precision': precision_score(y_test_classes, y_pred_classes, average='weighted'),
        'recall': recall_score(y_test_classes, y_pred_classes, average='weighted'),
        'f1': f1_score(y_test_classes, y_pred_classes, average='weighted'),
        'kappa': cohen_kappa_score(y_test_classes, y_pred_classes),
        'auc': roc_auc_score(y_test_classes, y_pred_proba, multi_class='ovr', average='weighted'),
        'confusion_matrix': confusion_matrix(y_test_classes, y_pred_classes)
    }
    
    # Print metrics
    for metric_name, metric_value in metrics.items():
        if metric_name != 'confusion_matrix':
            print(f"{metric_name.capitalize()}: {metric_value:.4f}")
    
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])
    
    return metrics

def plot_confusion_matrix(confusion_mat):
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        confusion_mat: Confusion matrix from sklearn
    """
    plt.figure(figsize=(10, 8), dpi=150)
    
    subject_labels = [f'S{i}' for i in range(1, 11)]
    
    plt.imshow(confusion_mat, cmap='gnuplot', extent=[0, 10, 0, 10])
    
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    
    # Set tick positions and labels
    plt.xticks(np.arange(0.5, 10.5), subject_labels, rotation=18)
    plt.yticks(np.arange(0.5, 10.5), subject_labels[::-1], rotation=18)
    
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.savefig('confusion_matrix.png')
    plt.show()

def main():
    """Main function to run the entire pipeline."""
    # Mount Google Drive if running in Colab
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        data_path = '/content/drive/MyDrive/My/EEG/Data/'
    except ImportError:
        # Not running in Colab, use local path
        data_path = './data/'
    
    # Load data
    X_train, X_test, y_train, y_test = load_data(data_path)
    
    # Build model
    model = build_model()
    model.summary()
    
    # Train model
    history = train_model(model, X_train, y_train, X_test, y_test)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Plot confusion matrix
    plot_confusion_matrix(metrics['confusion_matrix'])
    
    # Save model
    model.save('eeg_classifier_model.h5')
    print("Model saved as 'eeg_classifier_model.h5'")

if __name__ == "__main__":
    main()

