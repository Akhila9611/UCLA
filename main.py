import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from src.features.build_features import preprocess_data
from src.models.train_model import train_mlp_model
from src.models.predict_model import predict
import matplotlib.pyplot as plt

def main():
    # Load and preprocess data
    data = pd.read_csv('data/raw/Admission.csv')
    data['Admit_Chance'] = (data['Admit_Chance'] >= 0.8).astype(int)
    
    # Preprocess features and target
    X, y = preprocess_data(data)
    
    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model and save it
    model = train_mlp_model(X_train_scaled, y_train)

    # Make predictions on the training data
    ypred_train = model.predict(X_train_scaled)

    # Evaluate accuracy of the model on the training data
    train_accuracy = accuracy_score(y_train, ypred_train)
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

    # Make predictions on the test data
    ypred_test = model.predict(X_test_scaled)

    # Evaluate accuracy of the model on the test data
    test_accuracy = accuracy_score(y_test, ypred_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Confusion Matrix for test data
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, ypred_test)
    print(cm)

    # Save the trained model to a pickle file
    with open('src/models/mlp_model.pkl', 'wb') as file:
        pickle.dump(model, file)

    # Plotting loss curve
    loss_values = model.loss_curve_
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label='Loss', color='blue')
    plt.title('Loss Curve during Training')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
