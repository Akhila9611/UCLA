import pandas as pd

def preprocess_data(df):
    """
    Preprocess the data for modeling.
    """
    # One-hot encode categorical variables
    df['University_Rating'] = df['University_Rating'].astype('object')
    df['Research'] = df['Research'].astype('object')
    df = pd.get_dummies(df, columns=['University_Rating', 'Research'], dtype='int')

    # Drop unnecessary columns
    df = df.drop(['Serial_No'], axis=1)

    # Separate features and target
    X = df.drop(['Admit_Chance'], axis=1)
    y = df['Admit_Chance']
    
    return X, y
