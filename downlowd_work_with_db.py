import sqlite3
import pandas as pd

# Підключення до бази даних
conn = sqlite3.connect('sales_data.db')

# Завантаження даних у Pandas
df = pd.read_sql_query('SELECT * FROM sales', conn)
conn.close()

print(df)

#data_visualisation

import matplotlib.pyplot as plt

# Групуємо дані за продуктами
grouped = df.groupby('product')['quantity'].sum()

# Створюємо графік
grouped.plot(kind='bar', color=['blue', 'green', 'orange'])
plt.title('Продажі за продуктами')
plt.xlabel('Продукт')
plt.ylabel('Кількість проданих одиниць')
plt.show()


#Прогнозування продажів ШІ

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# Додаємо новий стовпець "total" (загальний дохід)
df['total'] = df['quantity'] * df['price']

# Перетворюємо дату в числовий формат
df['date'] = pd.to_datetime(df['date'])
df['date_numeric'] = df['date'].map(pd.Timestamp.toordinal)

# Вибираємо ознаки і цільову змінну
X = df[['date_numeric']]
y = df['total']

# Розділяємо дані на тренувальний і тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Створюємо модель і тренуємо її
model = LinearRegression()
model.fit(X_train, y_train)

# Прогнозуємо на тестових даних
y_pred = model.predict(X_test)

# Оцінюємо модель
print("Коефіцієнт детермінації (R^2):", model.score(X_test, y_test))

# Прогнозуємо на майбутнє
future_dates = pd.date_range(start='2024-01-08', periods=10)
future_dates_numeric = future_dates.map(pd.Timestamp.toordinal).to_numpy().reshape(-1, 1)
future_sales = model.predict(future_dates_numeric)

# Виводимо прогноз
for date, sales in zip(future_dates, future_sales):
    print(f"Дата: {date.date()}, Прогнозований дохід: {sales:.2f}")

# Експортуємо результати у файл Excel
output_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted Revenue': future_sales
})

output_df.to_excel('predicted_sales.xlsx', index=False, engine='openpyxl')
print("Прогнози збережено у файл predicted_sales.xlsx")
