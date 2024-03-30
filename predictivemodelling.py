import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Function to generate random sales data
def generate_data(num_samples):
    np.random.seed(42)
    data = {
        'Month': np.arange(1, num_samples + 1),
        'Sales': np.random.randint(100, 1000, size=num_samples)
    }
    return pd.DataFrame(data)

# Define the Streamlit app
def main():
    st.title('Demand Forecasting and Sales Trends Analysis')

    # Sidebar options
    st.sidebar.header('Options')
    num_samples = st.sidebar.slider('Number of Samples', min_value=10, max_value=100, value=50)
    model_type = st.sidebar.selectbox('Select Model Type', ['Linear Regression', 'Decision Tree'])

    # Generate random data
    df = generate_data(num_samples)

    # Display images
    st.image('sales.png', caption='Sales Trend', use_column_width=True)
    

    # Display random data
    st.subheader('Sales Data:')
    st.write(df)

    # Split data into features (X) and target variable (y)
    X = df[['Month']]
    y = df['Sales']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    if model_type == 'Linear Regression':
        model = LinearRegression()
    # Add other models here like Decision Tree, Time Series Analysis, etc.

    model.fit(X_train, y_train)
     
    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)

    # Display prediction results
    st.image('predicitve.png', caption='Predictive Modeling', use_column_width=True)
    st.subheader('Prediction Results:')
    st.write(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}))
    st.write("Mean Squared Error:", mse)

# Run the app
if __name__ == '__main__':
    main()
