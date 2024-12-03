# %% [markdown]
# Final Project | Option 3 | msw8gh | Phising Email Detector Neural Network

# %% [markdown]
# Import modules and load data from csv file

# %%
import pandas as pd
import numpy as np
import matplotlib as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import linear, relu, sigmoid
from tensorflow.keras.regularizers import l1_l2



df = pd.read_csv('phishing_dataset.csv')
df = df.sample(n=75000, random_state=None) #get random 75,000 sample from training data of 235,000

# %% [markdown]
# Seperate the feature and output columns. Debug features to ensure correctness

# %%
X = df.drop('label', axis=1) 
y = df['label']

print("Analyzing y\n1 = legimate email\n0 = phishing email")
print(y.value_counts())
print(f"There are {X.columns.size} features in the dataset")

# %% [markdown]
# Split the data into a 60-20-20 split using train_test_split from skLearn

# %%
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, shuffle=True)
X_cv, X_test, y_cv, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=84, shuffle=True)

print("Training set shape:", X_train.shape)
print("Validation set shape:", X_cv.shape)
print("Test set shape:", X_test.shape)

# %% [markdown]
# Function to apply tfidf (Term Frequency - Inverse Document Frequency) vectorization to features that are strings and converts them to numerical format. The string feature columns are deleted, and new features columns are added into the dataset that contain values from the vectorization. AI was used to assist in this function since it is a very complicated module that I have not worked in before

# %%
def apply_tfidf(X_train, X_cv, X_test, column, max_features=10):
    # Initialize TF-IDF vectorizer
    tfidf = TfidfVectorizer(max_features=max_features)

    # Fit the vectorizer only on the training set and transform all sets
    tfidf_train = tfidf.fit_transform(X_train[column].fillna(''))
    tfidf_cv = tfidf.transform(X_cv[column].fillna(''))
    tfidf_test = tfidf.transform(X_test[column].fillna(''))

    # Convert to DataFrame for easier integration with other features
    tfidf_train_df = pd.DataFrame(tfidf_train.toarray(), columns=[f"{column}_tfidf_{i}" for i in range(tfidf_train.shape[1])])
    tfidf_cv_df = pd.DataFrame(tfidf_cv.toarray(), columns=[f"{column}_tfidf_{i}" for i in range(tfidf_cv.shape[1])])
    tfidf_test_df = pd.DataFrame(tfidf_test.toarray(), columns=[f"{column}_tfidf_{i}" for i in range(tfidf_test.shape[1])])

    # Drop the original column and add the TF-IDF features
    X_train = X_train.drop(column, axis=1).reset_index(drop=True)
    X_cv = X_cv.drop(column, axis=1).reset_index(drop=True)
    X_test = X_test.drop(column, axis=1).reset_index(drop=True)

    X_train = pd.concat([X_train, tfidf_train_df], axis=1)
    X_cv = pd.concat([X_cv, tfidf_cv_df], axis=1)
    X_test = pd.concat([X_test, tfidf_test_df], axis=1)

    return X_train, X_cv, X_test



for feature in X_train.columns:
    if X_train[feature].dtype == 'object':  # Detect text columns
        X_train, X_cv, X_test = apply_tfidf(X_train, X_cv, X_test, feature, max_features=10)

print("Training set shape:", X_train.shape)
print("Validation set shape:", X_cv.shape)
print("Test set shape:", X_test.shape)

# %% [markdown]
# Use StandardScalar to scale each split independently

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  
X_cv_scaled = scaler.transform(X_cv)           
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# Make a function that creates the Keras neural network, with all parameters being model parameters to simplify testing

# %%
#initialize model
model = Sequential(
    [tf.keras.Input(shape=(X_train_scaled.shape[1],)),],  
) 

model.add(Dense(units=64, activation="relu", kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
model.add(Dropout(0.5))
model.add(Dense(units=32, activation="relu", kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
model.add(Dropout(0.5))

model.add(Dense(units=1, activation="sigmoid")) #output layer

model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])


modelHistory = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_cv_scaled, y_cv),
    epochs=10,
    batch_size=32,
    verbose=1
)

test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"{"Phishing Detection Model"}\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# %% [markdown]
# Evauluate model metrics

# %%
y_pred = model.predict(X_test_scaled)
y_pred = (y_pred > 0.5).astype(int)  #convert probabilities to 0 or 1

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
#----
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(f"True Positives: {cm[1,1]}\nFalse Positives: {cm[0,1]}\nTrue Negatives: {cm[0,0]}\nFalse Negatives: {cm[1,0]}\n")


