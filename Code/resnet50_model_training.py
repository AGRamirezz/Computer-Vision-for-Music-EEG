#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNet50 Model Training for EEG Classification

This script trains a ResNet50 model on EEG data for classification tasks.
The data is preprocessed and loaded as image-like representations of EEG signals.

Original notebook: ResNet50_Model_Training.ipynb
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import argparse
import gc


def parse_arguments():
    """
    Parse command-line arguments for the script.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='ResNet50 Model Training for EEG Classification')
    parser.add_argument('--data-path', type=str, default='/content/drive/MyDrive/EEG_DL/NMED_T/Xy_psd/',
                        help='Path to the directory containing data files')
    parser.add_argument('--epochs', type=int, default=125,
                        help='Number of epochs for training')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.008,
                        help='Learning rate for the optimizer')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results and model')
    parser.add_argument('--skip-visualization', action='store_true',
                        help='Skip data visualization')
    return parser.parse_args()


def load_data(data_path):
    """
    Load training and testing data from numpy files.
    
    Args:
        data_path (str): Path to the directory containing data files
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) - Training and testing data arrays and labels
    """
    print(f"Loading data from {data_path}...")
    
    # Load training and testing data
    X_train = np.load(os.path.join(data_path, "x_traing3.npy"))
    X_test = np.load(os.path.join(data_path, "x_testg3.npy"))
    y_train = np.load(os.path.join(data_path, "y_train2.npy"))
    y_test = np.load(os.path.join(data_path, "y_test2.npy"))
    
    print(f"X shapes: {X_train.shape}, {X_test.shape}")
    print(f"y shapes: {y_train.shape}, {y_test.shape}")
    
    return X_train, X_test, y_train, y_test


def visualize_data(X_train, num_examples=5):
    """
    Visualize examples from the training data as grayscale images.
    
    Args:
        X_train (numpy.ndarray): Training data array
        num_examples (int): Number of examples to visualize
    """
    print("Visualizing data examples...")
    
    font = {'family': 'Verdana',
            'color':  'black',
            'size': 13,
            }
    
    for i in range(num_examples):
        example = X_train[10+i,:,:,0]
        plt.figure(dpi=100)
        
        plt.yticks([0.5,25.5,50.5,75.5,100.5,124.5],['125','100','75','50','25','1'])
        plt.xticks([0.5,25.5,50.5,75.5,100.5,124.5],['1','25','50','75','100','125'])
        image = plt.imshow(example.T, cmap='Greys', interpolation='nearest')
        plt.ylabel('Channels', fontdict=font, labelpad=16)
        plt.xlabel('Samples', fontdict=font, labelpad=16)
        plt.title(f'EEG Data Example {i+1}')
        plt.tight_layout()
        
        # Save the figure if output directory exists
        if os.path.exists('results'):
            plt.savefig(f'results/eeg_example_{i+1}.png')
        
        plt.show()


def build_resnet50_model(input_shape=(125, 125, 3), num_classes=10):
    """
    Build and compile a ResNet50 model for classification.
    
    Args:
        input_shape (tuple): Shape of input data
        num_classes (int): Number of output classes
        
    Returns:
        tensorflow.keras.Model: Compiled ResNet50 model
    """
    print("Building ResNet50 model...")
    
    # Import ResNet50 model
    try:
        from classification_models.tfkeras import Classifiers
        resnet_class, preprocess_input = Classifiers.get('resnet50')
        base_model = resnet_class(input_shape=input_shape, weights='imagenet', include_top=True)
    except ImportError:
        print("Installing classification_models package...")
        import subprocess
        subprocess.check_call(["pip", "install", "git+https://github.com/qubvel/classification_models.git"])
        
        from classification_models.tfkeras import Classifiers
        resnet_class, preprocess_input = Classifiers.get('resnet50')
        base_model = resnet_class(input_shape=input_shape, weights='imagenet', include_top=True)
    
    # Modify the model for our classification task
    x = base_model.layers[-3].output
    output = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create the new model
    model = Model(base_model.input, output)
    
    return model


def train_model(model, X_train, y_train, X_test, y_test, learning_rate=0.008, epochs=125, batch_size=100):
    """
    Train the model on the provided data.
    
    Args:
        model (tensorflow.keras.Model): Model to train
        X_train (numpy.ndarray): Training data
        y_train (numpy.ndarray): Training labels
        X_test (numpy.ndarray): Testing data
        y_test (numpy.ndarray): Testing labels
        learning_rate (float): Learning rate for the optimizer
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        
    Returns:
        tensorflow.keras.callbacks.History: Training history
    """
    print(f"Training model with learning_rate={learning_rate}, epochs={epochs}, batch_size={batch_size}...")
    
    # Compile the model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['categorical_accuracy']
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test)
    )
    
    return history


def plot_training_history(history, save_path=None):
    """
    Plot the training and validation loss.
    
    Args:
        history (tensorflow.keras.callbacks.History): Training history
        save_path (str, optional): Path to save the plot
    """
    print("Plotting training history...")
    
    plt.figure(dpi=100)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.show()


def main():
    """
    Main function to run the ResNet50 model training pipeline.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data(args.data_path)
    
    # Visualize data if not skipped
    if not args.skip_visualization:
        visualize_data(X_train)
    
    # Build model
    model = build_resnet50_model(
        input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]),
        num_classes=y_train.shape[1]
    )
    
    # Print model summary
    model.summary()
    
    # Train model
    history = train_model(
        model, X_train, y_train, X_test, y_test,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Plot training history
    plot_training_history(
        history, 
        save_path=os.path.join(args.output_dir, 'training_history.png')
    )
    
    # Save model
    model.save(os.path.join(args.output_dir, 'resnet50_eeg_model.h5'))
    print(f"Model saved to {os.path.join(args.output_dir, 'resnet50_eeg_model.h5')}")
    
    # Clean up memory
    gc.collect()
    
    print("ResNet50 model training completed successfully!")


if __name__ == "__main__":
    main()



