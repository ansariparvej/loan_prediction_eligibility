import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import pickle

def main():
    df = pd.read_csv('/app/loan_eligibility_prediction/Q6.csv')
    print(df.columns)
    # Separate features and target variable
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']

    # Define the mapping for each categorical variable
    gender_mapping = {'Male': 0, 'Female': 1}
    married_mapping = {'No': 0, 'Yes': 1}
    dependents_mapping = {'0': 0, '1': 1, '2': 2, '3': 3}
    education_mapping = {'Not Graduate': 0, 'Graduate': 1}
    self_employed_mapping = {'No': 0, 'Yes': 1}
    property_area_mapping = {'Rural': 0, 'Semiurban': 1, 'Urban': 2}
    loan_status_mapping = {'N': 0, 'Y': 1}

    # Apply the mappings to the categorical variables
    X['Gender'] = X['Gender'].map(gender_mapping)
    X['Married'] = X['Married'].map(married_mapping)
    X['Dependents'] = X['Dependents'].map(dependents_mapping)
    X['Education'] = X['Education'].map(education_mapping)
    X['Self_Employed'] = X['Self_Employed'].map(self_employed_mapping)
    X['Property_Area'] = X['Property_Area'].map(property_area_mapping)
    y = y.map(loan_status_mapping)

    # List of Models:
    models = {
        'Logistic Regression': LogisticRegression(),
        'SVM': SVC(),
        'GradientBoost': GradientBoostingClassifier(),
        'KNN': KNeighborsClassifier()
    }

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Applying Standard Normalization method (Z-Score Equation):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Handling missing values
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)


    accuracy_scores = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores[model_name] = accuracy

    # Print accuracy scores
    for model_name, accuracy in accuracy_scores.items():
        print(f'{model_name}: {accuracy}')

    best_model_name = max(accuracy_scores, key=accuracy_scores.get)
    best_model = models[best_model_name]

    file_path = '/home/ali/anaconda3/Coding Playground/ML_Deployment/loan_eligibility_prediction/loan_eligibility_model.pkl'
    with open(file_path, 'wb') as file:
        pickle.dump(best_model, file)


if __name__ == '__main__':
    main()