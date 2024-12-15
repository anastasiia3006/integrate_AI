import sqlite3

conn = sqlite3.connect('sales_data.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS sales (
    id INTEGER PRIMARY KEY,
    date TEXT,
    product TEXT,
    quantity INTEGER,
    price REAL
)
''')

# Додаємо дані
data = [
    ('2024-01-01', 'Laptop', 3, 1500),
    ('2024-01-02', 'Smartphone', 10, 700),
    ('2024-01-03', 'Tablet', 5, 300),
    ('2024-01-04', 'Smartphone', 7, 700),
    ('2024-01-05', 'Laptop', 2, 1500),
    ('2024-01-06', 'Tablet', 8, 300),
    ('2024-01-07', 'Laptop', 1, 1500)
]

cursor.executemany('INSERT INTO sales (date, product, quantity, price) VALUES (?, ?, ?, ?)', data)
conn.commit()
conn.close()