# Task 1: Iris Flower Classification - CodeAlpha
# Dataset: Iris.csv with columns Id, SepalLengthCm, etc.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 1. Load your CSV file (upload Iris.csv in Colab first)
df = pd.read_csv('Iris.csv')
print(df.head())
print(df['Species'].value_counts())

# 2. Drop Id column
df = df.drop('Id', axis=1)

# 3. Split
X = df.iloc[:, :-1] # first 4 columns
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 6. Graph
sns.countplot(x='Species', data=df)
plt.title("Iris Species Count")
plt.show()

# 7. Save prediction example
test = [[5.1, 3.5, 1.4, 0.2]]
print(f"\nPrediction for {test}: {model.predict(test)[0]}")