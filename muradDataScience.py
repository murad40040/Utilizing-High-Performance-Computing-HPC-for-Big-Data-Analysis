#!/usr/bin/env python
# coding: utf-8

# In[4]:



# In[5]:


# =============================================================================
# Section 1: Imports and Setup
# =============================================================================
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

print("âœ… All libraries imported successfully.")


# In[7]:


# =============================================================================
# Section 2: Data Loading and Initial Exploration
# =============================================================================
print("\n[INFO] Starting data loading and exploration...")
start_load_time = time.time()

# --- For development, you might want to load a smaller sample ---
# df = pd.read_csv('flights.csv', nrows=500000) 
# --- For the final run on HPC, use the full dataset ---
df = pd.read_csv('flights.csv')

load_time = time.time() - start_load_time
print(f"Data loaded in {load_time:.2f} seconds.")
print(f"Dataset shape: {df.shape}")

# Display basic info and first 5 rows
print("\nDataset Info:")
df.info()
print("\nFirst 5 rows of the dataset:")
print(df.head())


# In[8]:


# =============================================================================
# Section 3: Data Preprocessing
# =============================================================================
print("\n[INFO] Starting data preprocessing...")

# 1. Create the binary target variable 'IS_DELAYED'
# A flight is considered "delayed" if it arrives more than 15 minutes late.
df['IS_DELAYED'] = (df['ARRIVAL_DELAY'] > 15).astype(int)

# 2. Drop columns that are not useful or have too many missing values
# For example, 'LATE_AIRCRAFT_DELAY', 'CANCELLATION_REASON' are mostly NaN.
# 'YEAR' is constant (2015), and 'FLIGHT_NUMBER' is just an identifier.
df.drop(columns=['YEAR', 'FLIGHT_NUMBER', 'TAIL_NUMBER', 'ARRIVAL_DELAY', 'CANCELLATION_REASON', 
                   'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY'], 
          inplace=True)

# 3. Select a smaller set of meaningful features for this example
# NOTE: For a more advanced analysis, you could one-hot encode 'AIRLINE', 'ORIGIN_AIRPORT', etc.
# But this would create a very wide dataset, so we'll stick to numeric features for this example.
features = ['MONTH', 'DAY', 'DAY_OF_WEEK', 'SCHEDULED_DEPARTURE', 'SCHEDULED_TIME', 'DISTANCE', 'SCHEDULED_ARRIVAL']
target = 'IS_DELAYED'

# 4. Handle remaining missing values
# We'll drop rows where any of our selected features or the target are missing.
df.dropna(subset=features + [target], inplace=True)
print(f"Shape after dropping NaNs: {df.shape}")


# 5. Create the final feature matrix (X) and target vector (y)
X = df[features]
y = df[target]

print("âœ… Preprocessing complete.")
print(f"Final feature shape (X): {X.shape}")
print(f"Final target shape (y): {y.shape}")


# In[9]:


# =============================================================================
# Section 4: Data Visualization
# =============================================================================
print("\n[INFO] Generating visualizations...")

# 1. Visualize the balance of the target variable
plt.figure(figsize=(8, 6))
sns.countplot(x=y)
plt.title('Distribution of Delayed vs. On-Time Flights', fontsize=16)
plt.xlabel('Flight Status (0: On-Time, 1: Delayed)')
plt.ylabel('Count')
plt.savefig('target_distribution.png') # Saves the plot to a file
#plt.show()
print("Saved target_distribution.png")

# 2. Correlation Heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = X.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix of Features', fontsize=16)
plt.savefig('correlation_heatmap.png') # Saves the plot to a file
#plt.show()
print("Saved correlation_heatmap.png")


# In[10]:


# =============================================================================
# Section 5: Model Training and Evaluation
# =============================================================================
print("\n[INFO] Starting model training and evaluation...")

# 1. Split data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# 2. Train and Evaluate Model 1: Logistic Regression
print("\n--- Training Logistic Regression ---")
# Parameter justification: C=1.0 is a standard baseline. max_iter is increased to ensure convergence on large data.
log_reg = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
start_time = time.time()
log_reg.fit(X_train, y_train)
log_reg_time = time.time() - start_time
y_pred_log = log_reg.predict(X_test)

print(f"Logistic Regression training time: {log_reg_time:.2f} seconds")
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_log):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_log))


# 3. Train and Evaluate Model 2: Random Forest Classifier
print("\n--- Training Random Forest Classifier ---")
# Parameter justification: n_estimators=100 is a good balance of performance and speed. 
# max_depth=10 prevents overfitting. n_jobs=-1 uses all available CPU cores on the HPC node.
rand_forest = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
start_time = time.time()
rand_forest.fit(X_train, y_train)
rf_time = time.time() - start_time
y_pred_rf = rand_forest.predict(X_test)

print(f"Random Forest training time: {rf_time:.2f} seconds")
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))



# In[11]:


# =============================================================================
# Section 6: Final Comparison
# =============================================================================
print("\n[INFO] Final Model Comparison:")
print("="*40)
print(f"| {'Metric':<20} | {'Logistic Regression':<20} |")
print(f"| {'-'*20} | {'-'*20} |")
print(f"| {'Training Time (s)':<20} | {log_reg_time:<20.2f} |")
print(f"| {'Accuracy':<20} | {accuracy_score(y_test, y_pred_log):<20.4f} |")
print("="*40)
print(f"| {'Metric':<20} | {'Random Forest':<20} |")
print(f"| {'-'*20} | {'-'*20} |")
print(f"| {'Training Time (s)':<20} | {rf_time:<20.2f} |")
print(f"| {'Accuracy':<20} | {accuracy_score(y_test, y_pred_rf):<20.4f} |")
print("="*40)

print("\nðŸš€ Project script finished.")


# In[ ]:




