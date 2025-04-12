import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def train_mlp_model(X_train, y_train, model_save_path='src/models/mlp_model.pkl'):
    """
    Train a Multi-layer Perceptron Classifier (MLP) model and save it to a pickle file.
    """
    # Initialize MLP Classifier
    model = MLPClassifier(hidden_layer_sizes=(3, 3), batch_size=50, max_iter=200, random_state=123)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model accuracy on training data
    ypred_train = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, ypred_train)
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    
    # Save the trained model to pickle
    with open(model_save_path, 'wb') as file:
        pickle.dump(model, file)
    
    return model
