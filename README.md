# 🌸 Iris Flower Classification - CodeAlpha Internship

## Task 1 - Data Science Internship

### 📌 Objective
Classify Iris flowers into 3 species (Setosa, Versicolor, Virginica) based on Sepal and Petal measurements.

### 📂 Dataset
- **Source:** Iris.csv (150 samples)
- **Features:** SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm
- **Target:** Species
- **Distribution:** 50 samples per species (Balanced)

### 🛠️ Tech Stack
- Python
- Pandas, Numpy
- Matplotlib, Seaborn
- Scikit-learn (RandomForestClassifier)

### ⚙️ Steps Performed
1. Data Loading & Exploration
2. Data Cleaning (Removed Id column)
3. EDA - Species count plot
4. Train-Test Split (80-20)
5. Model Training - Random Forest
6. Evaluation - Accuracy, Classification Report, Confusion Matrix

### 📊 Results
- **Accuracy:** 100% (1.00)
- **Model:** RandomForestClassifier (n_estimators=100)
- All 3 species classified correctly.

### 🚀 How to Run
```bash
pip install pandas scikit-learn matplotlib seaborn
python c2.py
