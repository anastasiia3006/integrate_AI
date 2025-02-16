import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np


def fetch_sales_data(db_path: str, query: str = 'SELECT * FROM sales') -> pd.DataFrame:
    """
    Connects to the database and loads data from the sales table.
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)
    return df


def plot_sales_by_product(df: pd.DataFrame):
    """
    Creates a bar chart displaying the total quantity sold for each product.
    """
    # Group data by product and calculate the sum of quantities sold
    grouped = df.groupby('product')['quantity'].sum()
    # Generate a color palette based on the number of products
    colors = plt.cm.viridis_r(np.linspace(0, 1, len(grouped)))
    grouped.plot(kind='bar', color=colors)
    plt.title('Sales by Product')
    plt.xlabel('Product')
    plt.ylabel('Quantity Sold')
    plt.tight_layout()
    plt.show()


def predict_future_sales(df: pd.DataFrame, future_start: str = '2024-01-08', periods: int = 10):
    """
    Forecasts future sales using linear regression.
    """
    # Create a new column for total revenue
    df['total'] = df['quantity'] * df['price']

    # Convert the date column to datetime format and then to a numeric format
    df['date'] = pd.to_datetime(df['date'])
    df['date_numeric'] = df['date'].map(pd.Timestamp.toordinal)

    # Define the feature and target variables
    X = df[['date_numeric']]
    y = df['total']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model using the coefficient of determination (R^2)
    score = model.score(X_test, y_test)
    print(f"Coefficient of Determination (R^2): {score:.2f}")

    # Generate future dates for forecasting
    future_dates = pd.date_range(start=future_start, periods=periods)
    future_dates_numeric = future_dates.map(pd.Timestamp.toordinal).to_numpy().reshape(-1, 1)
    future_sales = model.predict(future_dates_numeric)

    # Print the forecasted revenue for each future date
    for date, sales in zip(future_dates, future_sales):
        print(f"Date: {date.date()}, Predicted Revenue: {sales:.2f}")

    # Export the forecast results to an Excel file
    output_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Revenue': future_sales
    })
    output_df.to_excel('predicted_sales.xlsx', index=False, engine='openpyxl')
    print("Forecast saved to predicted_sales.xlsx")


def main():
    db_path = 'sales_data.db'

    # Fetch sales data from the database
    df = fetch_sales_data(db_path)
    print("Fetched data:")
    print(df)

    # Visualize sales by product
    plot_sales_by_product(df)

    # Forecast future sales and export results
    predict_future_sales(df)


if __name__ == '__main__':
    main()
