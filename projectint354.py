

#PROJECT:- **Glioma Grading Clinical and Mutation Features**
"""

# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load your dataset
data = pd.read_csv('/content/TCGA_InfoWithGrade.csv')

# Display the first few rows of the dataset
print(data.head())

data.head()

# Display the number of rows and columns
print("Number of rows:", data.shape[0])
print("Number of columns:", data.shape[1])

# Display the shape of the DataFrame (number of rows and columns)
print("\nShape of the DataFrame (rows, columns):", data.shape)

print("\nSummary statistics of the DataFrame:")
data.describe()

data.columns

data.dtypes

data.isna().sum()

# Convert all columns to float64 data type
data = data.astype(float)
data.dtypes

data.isnull().sum()

# Assuming 'Category' is the categorical variable to be label encoded
label_encoder = LabelEncoder()

data['Grade'] = label_encoder.fit_transform(data['Grade'])
print(data['Grade'])

# Split the data into features (X) and target variable (y)
X = data.drop('Grade', axis=1)
y = data['Grade']               # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the Logistic Regression Model

model = LogisticRegression()
print(model)

# Fit the model to the training data
model.fit(X_train, y_train)

# Model Evaluation

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

"""ALL Steps

"""

# Step 1: Read Dataset
import pandas as pd

# Assuming you have the dataset in a CSV file named 'glioma_data.csv'
df = pd.read_csv('/content/TCGA_InfoWithGrade.csv')

print(df.head())

df.head()

df.info()

# Step 2: Data Preprocessing
# Assuming 'Grade' is the target value and other features are independent variables
X = df.drop('Grade', axis=1)
y = df['Grade']
y.shape

# One-hot encode categorical variables
X = pd.get_dummies(X)
print(X.head(2))
print(X.shape)

# Step 3: Apply minimum 3 classification/regression models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    "Random Forest": RandomForestClassifier(), "Support Vector Machine": SVC(), "Logistic Regression": LogisticRegression()
}

# Training and evaluation
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    score = model.score(X_test_scaled, y_test)
    results[name] = score

# Step 4: Apply Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

# Define hyperparameters grid for each model
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20]
}

param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly']
}

param_grid_lr = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2']
}

# Initialize GridSearchCV for each model
grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf)
grid_search_svm = GridSearchCV(SVC(), param_grid_svm)
grid_search_lr = GridSearchCV(LogisticRegression(), param_grid_lr)

# Fit the models with hyperparameter tuning
grid_search_rf.fit(X_train_scaled, y_train)

grid_search_svm.fit(X_train_scaled, y_train)

grid_search_lr.fit(X_train_scaled, y_train)

# Step 5: Compare the results
best_results = {
    "Random Forest": grid_search_rf.best_score_,
    "Support Vector Machine": grid_search_svm.best_score_,
    "Logistic Regression": grid_search_lr.best_score_
}

print("Results before hyperparameter tuning:")
print(results);



# print("Best results after hyperparameter tuning:", best_results)

print("Best results after hyperparameter tuning:")
print(grid_search_rf.best_score_)
print(grid_search_svm.best_score_)
print(grid_search_lr.best_score_)
print("Support Vector Machine model performs the best after hyperparameter tuning based on the provided results.")

"""**import precision_score, recall_score, f1_score, **"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert one-hot encoded labels to single column
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Initialize models
models = {
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(probability=True),
    "Logistic Regression": LogisticRegression()
}

# Initialize dictionaries to store metrics
precision_scores = {}
recall_scores = {}
f1_scores = {}

# Training, evaluation, and metric calculation for each model
for name, model in models.items():
    # Train the model
    model.fit(X_train_scaled, y_train_encoded)

    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)

    # Calculate metrics
    precision_scores[name] = precision_score(y_test_encoded, y_pred, average='weighted')
    recall_scores[name] = recall_score(y_test_encoded, y_pred, average='weighted')
    f1_scores[name] = f1_score(y_test_encoded, y_pred, average='weighted')

# Print the metrics
for name in models.keys():
    print(f"Metrics for {name}:")
    print("Precision:", precision_scores[name])
    print("Recall:", recall_scores[name])
    print("F1-Score:", f1_scores[name])

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert one-hot encoded labels to single column
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Initialize models
models = {
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(probability=True),
    "Logistic Regression": LogisticRegression()
}

# Initialize dictionaries to store metrics
precision_scores = {}
recall_scores = {}
f1_scores = {}

# Training, evaluation, and metric calculation for each model
for name, model in models.items():
    # Train the model
    model.fit(X_train_scaled, y_train_encoded)

    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)

    # Calculate metrics
    precision_scores[name] = precision_score(y_test_encoded, y_pred, average='weighted')
    recall_scores[name] = recall_score(y_test_encoded, y_pred, average='weighted')
    f1_scores[name] = f1_score(y_test_encoded, y_pred, average='weighted')



# Print the metrics
for name in models.keys():
    print(f"Metrics for {name}:")
    print("Precision:", precision_scores[name])
    print("Recall:", recall_scores[name])
    print("F1-Score:", f1_scores[name])
    print()

import matplotlib.pyplot as plt

# Define models and corresponding metric scores
models_metrics = {
    "Precision": precision_scores,
    "Recall": recall_scores,
    "F1-Score": f1_scores
}

# Plot bar charts for each metric
for metric, scores in models_metrics.items():
    plt.figure(figsize=(8, 6))
    plt.bar(scores.keys(), scores.values())
    plt.title(f"{metric} Across Different Models")
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()

import matplotlib.pyplot as plt

# Combine precision, recall, and F1 scores for each model into a single DataFrame
metrics_df = pd.DataFrame({'Precision': precision_scores, 'Recall': recall_scores, 'F1-Score': f1_scores})

# Plotting
metrics_df.plot(kind='bar', figsize=(10, 6))
plt.title('Model Performance Metrics')
plt.xlabel('Model')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend(title='Metric')
plt.tight_layout()
plt.show()

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Assuming precision_scores, recall_scores, and f1_scores are dictionaries containing metric scores
# Convert metric scores into a DataFrame
metrics_df = pd.DataFrame({"Precision": precision_scores, "Recall": recall_scores, "F1-Score": f1_scores})

# Calculate correlation matrix
correlation_matrix = metrics_df.corr()

print(correlation_matrix)

import seaborn as sns
import matplotlib.pyplot as plt

# Assuming correlation_matrix is the correlation matrix calculated from metrics_df
# Plot correlation matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.6)
plt.title("Correlation Matrix of Metrics")
plt.xlabel("Metrics")
plt.ylabel("Metrics")
plt.show()

import seaborn as sns
# Plot box plot to visualize distribution of metric scores
plt.figure(figsize=(10, 6))
sns.boxplot(data=metrics_df)
plt.title(" BOX PLOT MODELS REPRESENTATION ")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# import seaborn as sns

# # Convert metric scores into a DataFrame
# metrics_df = pd.DataFrame({"Precision": precision_scores, "Recall": recall_scores, "F1-Score": f1_scores})

# # Pairplot to visualize pairwise relationships between metrics
# sns.pairplot(metrics_df)
# plt.suptitle('Pairplot of Metrics', y=1.02)
# # plt.show()

# Histogram to visualize the distribution of each metric
plt.figure(figsize=(10, 6))
plt.hist(metrics_df["Precision"], bins=10, alpha=0.5, label='Precision')
plt.hist(metrics_df["Recall"], bins=10, alpha=0.5, label='Recall')
plt.hist(metrics_df["F1-Score"], bins=10, alpha=0.5, label='F1-Score')
plt.title('Histogram of Metrics')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.legend()
plt.show()

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming your dataset is stored in a DataFrame called df
corr = np.corrcoef(df.values.T)

# Plot correlation matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.xlabel("Features")
plt.ylabel("Features")
plt.show()

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming your dataset is stored in a DataFrame called df
# Select the first 5 columns
df_subset = df.iloc[:, :5]

# Calculate correlation matrix
corr = np.corrcoef(df_subset.values.T)

# Plot correlation matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of First 5 Columns")
plt.xlabel("Features")
plt.ylabel("Features")
plt.show()