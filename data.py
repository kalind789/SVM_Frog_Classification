import pandas as pd
import sklearn

def get_data(dataset):
    # Load the dataset, skip the first row because it's useless
    df = pd.read_csv('dataset.csv', delimiter=';', skiprows=1)

    # The ID does not really do anything for the 
    df = df.drop('ID', axis=1)
    
    # We have some categorical data in the dataset, we must turn this into useful information. 
    
    # Set the target to the fire bellied toad
    target = df['Fire-bellied toad']
    # Set anything other than the fire bellied toad to the features
    features = df.drop(['Fire-bellied toad'], axis=1)


if __name__ == '__main__':
    dataset = 'dataset.csv'
    get_data(dataset=dataset)