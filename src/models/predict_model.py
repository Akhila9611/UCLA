def predict(model, X_input_scaled):
    """
    Make predictions using the trained model on new input data.
    """
    predictions = model.predict(X_input_scaled)
    return predictions
