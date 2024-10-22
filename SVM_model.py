import pandas
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from data import get_data

def apply_model_linear(X_train, y_train):
    # Initialize the linear models with different slack values
    linear_model_C_1 = SVC(kernel='linear', C=1.0)
    linear_model_C_2 = SVC(kernel='linear', C=2.0)
    linear_model_C_0_5 = SVC(kernel='linear', C=0.5)

    # Fit the models
    linear_model_C_1.fit(X_train, y_train)
    linear_model_C_2.fit(X_train, y_train)
    linear_model_C_0_5.fit(X_train, y_train)

    # Cross-validate the models
    linear_model_scores_C1 = cross_val_score(linear_model_C_1, X_train, y_train, cv=5, scoring='f1')
    linear_model_scores_C2 = cross_val_score(linear_model_C_2, X_train, y_train, cv=5, scoring='f1')
    linear_model_scores_C_05 = cross_val_score(linear_model_C_0_5, X_train, y_train, cv=5, scoring='f1')

    return linear_model_scores_C1, linear_model_scores_C2, linear_model_scores_C_05

if __name__ == '__main__':
    dataset = 'dataset.csv'

    # Get the split data
    X_train, X_test, y_train, y_test = get_data(dataset)

    # Apply the model on the training data
    C1_score, C2_score, C05_score = apply_model_linear(X_train, y_train)

    # Print the cross-validation scores for each model
    print("Cross-Validation Scores for C=1.0:", C1_score)
    print("Cross-Validation Scores for C=2.0:", C2_score)
    print("Cross-Validation Scores for C=0.5:", C05_score)
