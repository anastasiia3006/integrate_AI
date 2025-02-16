This code is designed for the analysis and forecasting of sales data stored in a SQLite database. It consists of three primary functions:
    Data Retrieval:
    The fetch_sales_data function connects to the SQLite database and retrieves data from the 'sales' table.
    Data Visualization:
    The plot_sales_by_product function groups the sales data by product and generates a bar chart that displays the total quantity sold for each product.
    Forecasting:
    The predict_future_sales function performs sales forecasting using linear regression. It converts date values into a numeric format, splits the data into training and test sets, evaluates the model's performance, predicts future revenue, and exports the forecasted results to an Excel file.
