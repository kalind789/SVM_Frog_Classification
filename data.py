import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import time


def get_data(dataset):
    # Load the dataset, skip the first row because it's useless
    df = pd.read_csv('dataset.csv', delimiter=';', skiprows=1)
    print(df.shape)

    # The ID does not really do anything for the 
    df = df.drop('ID', axis=1)
    
    # We have some categorical data in the dataset, we must turn this into useful information. 
    start = time.time()
    df_encoded = pd.get_dummies(df, columns=['Motorway', 'TR', 'VR', 'UR', 'FR', 'OR', 'RR', 'BR', 'MR', 'CR'])
    print("Encoding time:", time.time() - start)


    # Set the target to the fire bellied toad
    target = df_encoded['Fire-bellied toad']

    # Set anything other than the fire bellied toad to the features
    features = df_encoded.drop(['Fire-bellied toad'], axis=1)

    # Split the data into training and testing sets
    features_train, features_test, targets_train, targets_test = train_test_split(features, target, test_size=0.2, random_state=42)

    return features_train, features_test, targets_train, targets_test