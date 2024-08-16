import pandas as pd
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Loading Titanic dataset
file_path = 'Titanic-Dataset.csv'
titanic_data = pd.read_csv("Titanic-Dataset.csv")

# Drop irrelevant columns
titanic_data = titanic_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

# Handle missing values
imputer = SimpleImputer(strategy='mean')
titanic_data['Age'] = imputer.fit_transform(titanic_data[['Age']])
titanic_data['Embarked'] = titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0])

# Convert categorical variables to numerical
label_encoder = LabelEncoder()
titanic_data['Sex'] = label_encoder.fit_transform(titanic_data['Sex'])
titanic_data['Embarked'] = label_encoder.fit_transform(titanic_data['Embarked'])

# Define features and target variable
X = titanic_data.drop(columns=['Survived'])
y = titanic_data['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'titanic_survival_model.pkl')

# Load the model
model = joblib.load('titanic_survival_model.pkl')

# Streamlit app
st.title('Titanic Survival Prediction')

# User input
pclass = st.selectbox('Passenger Class', [1, 2, 3])
sex = st.selectbox('Sex', ['Male', 'Female'])
age = st.slider('Age', 0, 100, 25)
sibsp = st.slider('Number of Siblings/Spouses Aboard', 0, 8, 0)
parch = st.slider('Number of Parents/Children Aboard', 0, 6, 0)
fare = st.slider('Fare', 0.0, 520.0, 50.0)
embarked = st.selectbox('Port of Embarkation', ['C', 'Q', 'S'])

# Convert user input to dataframe
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [1 if sex == 'male' else 0],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [label_encoder.transform([embarked])[0]]
})

# Make prediction
if st.button('Predict'):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.write(f'Survived: {"Yes" if prediction[0] == 1 else "No"}')
    st.write(f'Probability of Survival: {prediction_proba[0][1]:.2f}')
