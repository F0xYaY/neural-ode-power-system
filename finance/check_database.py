import sqlite3
import pandas as pd

# 连接数据库
conn = sqlite3.connect('quant_database.db')

# 查看总记录数
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM daily_price')
total_records = cursor.fetchone()[0]
print(f'总记录数: {total_records}')

# 查看股票数量
cursor.execute('SELECT DISTINCT stock_code FROM daily_price')
codes = cursor.fetchall()
print(f'股票数量: {len(codes)}')
print('股票代码:', [c[0] for c in codes])

# 查看每个股票的数据条数
print('\n各股票数据条数:')
for code in codes:
    cursor.execute('SELECT COUNT(*) FROM daily_price WHERE stock_code = ?', (code[0],))
    count = cursor.fetchone()[0]
    print(f'  {code[0]}: {count} 条')

# 查看前5条记录
print('\n前5条记录:')
df = pd.read_sql_query('SELECT * FROM daily_price ORDER BY trade_date DESC LIMIT 5', conn)
print(df)

# 查看日期范围
cursor.execute('SELECT MIN(trade_date), MAX(trade_date) FROM daily_price')
date_range = cursor.fetchone()
print(f'\n日期范围: {date_range[0]} 至 {date_range[1]}')

conn.close()
