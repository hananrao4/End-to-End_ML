from flask import Flask, render_template, request
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Generate random data for Cholesterol Test
cholesterol_data = {
    'Age': [random.randint(30, 70) for _ in range(100)],
    'Diabetes History': [random.choice(['Yes', 'No']) for _ in range(100)],
    'Blood Pressure History': [random.choice(['Normal', 'High', 'Elevated']) for _ in range(100)],
    'Total Cholesterol': [random.randint(150, 250) for _ in range(100)],
    'HDL Cholesterol': [random.randint(40, 70) for _ in range(100)],
    'LDL Cholesterol': [random.randint(90, 180) for _ in range(100)],
    'Triglycerides': [random.randint(110, 220) for _ in range(100)],
    'Status': [random.choice(['Normal', 'Elevated', 'High']) for _ in range(100)],
    'Should go for ECG': [random.choice(['Yes', 'No']) for _ in range(100)],
}

cholesterol_df = pd.DataFrame(cholesterol_data)

# Generate random data for Liver Function Test
lft_data = {
    'Age': [random.randint(30, 70) for _ in range(100)],
    'Blood Pressure History': [random.choice(['Normal', 'High', 'Elevated']) for _ in range(100)],
    'Diabetes History': [random.choice(['Yes', 'No']) for _ in range(100)],
    'Alkaline Phosphatase (ALP)': [random.randint(50, 150) for _ in range(100)],
    'Aspartate Aminotransferase (AST)': [random.randint(10, 50) for _ in range(100)],
    'Alanine Aminotransferase (ALT)': [random.randint(10, 40) for _ in range(100)],
    'Total Bilirubin': [random.uniform(0.1, 1.2) for _ in range(100)],
    'Direct Bilirubin': [random.uniform(0.05, 0.5) for _ in range(100)],
    'Albumin': [random.uniform(3.5, 5.5) for _ in range(100)],
    'Status': [random.choice(['Normal', 'Abnormal']) for _ in range(100)],
    'Should go for LFT': [random.choice(['Yes', 'No']) for _ in range(100)],
}

lft_df = pd.DataFrame(lft_data)

# Placeholder for the machine learning model
model = RandomForestClassifier()

@app.route('/')
def index():
    return render_template('index_2.html')

@app.route('/predict', methods=['POST'])
def predict():
    test_type = request.form['test_type']
    
    if test_type == 'cholesterol':
        features = cholesterol_df.drop(['Status', 'Should go for ECG'], axis=1)
        target = cholesterol_df['Status']
    elif test_type == 'lft':
        features = lft_df.drop(['Status', 'Should go for LFT'], axis=1)
        target = lft_df['Status']
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    return render_template('prediction.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
    
