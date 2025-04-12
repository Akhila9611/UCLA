import pandas as pd

def load_data(file_path='data/raw/Admission.csv'):
    """
    Load the dataset from the provided file path.
    """
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    """
    Clean the dataset by converting the target variable to binary and dropping unnecessary columns.
    """
    # Convert target variable 'Admit_Chance' into categorical
    data['Admit_Chance'] = (data['Admit_Chance'] >= 0.8).astype(int)
    
    # Drop unnecessary columns
    data = data.drop(['Serial_No'], axis=1)
    # get unique values for LOR
    data['CGPA'].unique()
    # Convert categorical variables to object type for encoding
    data['University_Rating'] = data['University_Rating'].astype('object')
    data['Research'] = data['Research'].astype('object')
    data.info()
    return data
