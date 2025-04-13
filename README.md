# UCLA Neural Networks Solution

Link to app: "https://uclaapp.streamlit.app/"


## Project Overview

This project focuses on solving a machine learning problem using **neural networks**. The goal is to preprocess data, train models, and visualize results effectively. The project involves creating a neural network-based model for predicting the admission chance of applicants based on various features. The model is designed to evaluate performance metrics, with a focus on achieving high accuracy.

### **Goal**:
- The goal of this project is to build and train a neural network model that predicts the **admission chance** of applicants to universities based on their GRE scores, TOEFL scores, CGPA, and other factors.
- The model will be trained using a dataset and evaluated for accuracy and performance metrics to meet a specific target threshold (e.g., accuracy > 80%).

### **Machine Learning Task**:
- **Task Type**: (Classification or Regression) *(You can specify this based on your use case)*  
- **Target Variable**: `Target_Column_Name` *(Replace with actual target)*  
- **Success Criteria**: High accuracy or low error depending on task type.

---

## Files in the Project

1. **`preprocessing.py`**:
   - Contains functions for **data cleaning**, **handling missing values**, **encoding categorical variables**, and **normalizing the data**.
   
2. **`train_model.py`**:
   - Loads the dataset, preprocesses the data, trains the **neural network model**, and saves the trained model to a file (`neural_network_model.pkl`).
   
3. **`evaluate_model.py`**:
   - Evaluates the trained model's performance using various metrics like accuracy, loss, and mean squared error (MSE).
   
4. **`streamlit_app.py`**:
   - This is a **Streamlit app** that provides an interactive user interface to make predictions using the trained model. Users can input relevant features and get predictions for the admission chance.

---

## Dataset

The dataset used for this project contains the following columns:

- **Serial_No**: Unique identifier for each applicant.
- **GRE_Score**: GRE score of the applicant.
- **TOEFL_Score**: TOEFL score of the applicant.
- **University_Rating**: Rating of the university (1-5 scale).
- **SOP**: Statement of Purpose strength (1-5 scale).
- **LOR**: Letter of Recommendation strength (1-5 scale).
- **CGPA**: Cumulative Grade Point Average of the applicant.
- **Research**: Whether the applicant has research experience (1 = Yes, 0 = No).
- **Admit_Chance**: Target variable (Admission chance as a continuous value between 0 and 1).

---

## Steps to Run the Project

### **1. Install Dependencies**:
Ensure you have the required Python packages installed. You can install the necessary libraries using `pip`:

```bash
pip install -r requirements.txt
