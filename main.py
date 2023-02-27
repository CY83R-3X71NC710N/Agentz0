
#!/usr/bin/env python
# Copyright 2023 CY83R-3X71NC710N

# Import statements
import os
import sys
import time
import numpy as np
import pandas as pd
import scipy as sp
import sklearn as sk
import tensorflow as tf
import nltk as nl

# Main code
def main():
    # Load data
    data = pd.read_csv('malicious_code.csv')
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    # Split data into training and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    # Feature scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Train the model
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'linear', random_state = 0)
    classifier.fit(X_train, y_train)
    
    # Make predictions
    y_pred = classifier.predict(X_test)
    
    # Evaluate the model
    from sklearn.metrics import confusion_matrix, accuracy_score
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print('Accuracy: ' + str(accuracy_score(y_test, y_pred)))
    
    # Natural language processing
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    words = word_tokenize(data)
    words = [w for w in words if w not in stopwords.words('english')]
    words = [ps.stem(w) for w in words]
    print(words)
    
    # TensorFlow
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5)
    model.evaluate(X_test, y_test)
    
    # GUI development
    import tkinter as tk
    from tkinter import ttk
    window = tk.Tk()
    window.title("Malicious Code Analyzer")
    window.geometry("400x400")
    label = ttk.Label(window, text = "Malicious Code Analyzer")
    label.pack()
    window.mainloop()
    
# Finishing touches
if __name__ == "__main__":
    main()
