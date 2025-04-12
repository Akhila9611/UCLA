import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_loss_curve(model, save_path='UCLA/results/loss_curve.png'):
    """
    Plot the loss curve of the trained model and save it as a PNG file.
    """
    loss_values = model.loss_curve_
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label='Loss', color='blue')
    plt.title('Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Save the plot to a file
    plt.savefig(save_path)  # Save to the specified path
    plt.close()  # Close the plot to avoid overlapping with future plots

def plot_confusion_matrix(y_true, y_pred, save_path='UCLA/results/confusion_matrix.png'):
    """
    Plot and save the confusion matrix to a file.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Admitted', 'Admitted'], yticklabels=['Not Admitted', 'Admitted'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Save the plot to a file
    plt.savefig(save_path)  # Save to the specified path
    plt.close()  # Close the plot to avoid overlapping with future plots
