import sqlite3
import akshare as ak
import pandas as pd
import time
from datetime import datetime

# 1. 连接数据库 (如果不存在会自动创建)
conn = sqlite3.connect('quant_database.db')
cursor = conn.cursor()

# 2. 创建表结构
create_table_sql = '''
CREATE TABLE IF NOT EXISTS daily_price (
    trade_date TEXT,
    stock_code TEXT,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    amount REAL,
    PRIMARY KEY (trade_date, stock_code)
);
'''
cursor.execute(create_table_sql)
conn.commit()

# 3. 获取股票列表 (这里只演示前 5 只股票，实际使用去掉 .head(5))
stock_info = ak.stock_zh_a_spot_em()
stock_codes = stock_info['代码'].tolist()[:5] 

print(f"准备下载 {len(stock_codes)} 只股票的历史数据...")

# 4. 循环下载并存储
for code in stock_codes:
    try:
        print(f"正在下载: {code}")
        # 获取后复权数据 (Back-adjusted)，适合回测
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date="20200101", adjust="hfq")
        
        if df.empty:
            continue
            
        # 数据清洗：重命名列以匹配数据库字段
        df = df[['日期', '开盘', '最高', '最低', '收盘', '成交量', '成交额']]
        df.columns = ['trade_date', 'open', 'high', 'low', 'close', 'volume', 'amount']
        df['stock_code'] = code
        
        # 写入数据库 (如果存在则替换/忽略)
        df.to_sql('daily_price', conn, if_exists='append', index=False)
        
        # 稍微暂停一下，防止被封 IP
        time.sleep(0.5)
        
    except Exception as e:
        print(f"下载 {code} 失败: {e}")

# 5. 创建索引 (关键！能让查询速度快 100 倍)
print("正在创建索引...")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_code_date ON daily_price (stock_code, trade_date)")
conn.commit()
conn.close()

print("数据库初始化完成！")
