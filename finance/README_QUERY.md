# 股票数据查询和可视化工具使用指南

## 功能特性

- 📊 **价格走势图**: 显示股票收盘价和最高最低价区间
- 📈 **K线图**: 传统蜡烛图，显示开盘、收盘、最高、最低价
- 📉 **多股票对比**: 对比多只股票的收益率
- 📋 **统计信息**: 查看股票的基本统计数据
- 💾 **图表保存**: 支持将图表保存为PNG文件

## 使用方法

### 1. 命令行使用

#### 列出所有股票代码
```bash
python query_visualize.py --list
```

#### 查看股票统计信息
```bash
python query_visualize.py --code 300391 --stats
```

#### 绘制价格走势图
```bash
python query_visualize.py --code 300391 --type price
```

#### 绘制K线图（最近60天）
```bash
python query_visualize.py --code 300391 --type candlestick --days 60
```

#### 指定日期范围
```bash
python query_visualize.py --code 300391 --start 2024-01-01 --end 2024-12-31
```

#### 保存图表
```bash
python query_visualize.py --code 300391 --type price --save chart.png
```

#### 多股票对比
```bash
python query_visualize.py --code 300391,301200 --type compare
```

### 2. Python代码使用

```python
from query_visualize import StockQueryVisualizer

# 初始化
viz = StockQueryVisualizer()

# 列出所有股票
codes = viz.list_stocks()

# 查询股票数据
df = viz.query_stock('300391', start_date='2024-01-01', end_date='2024-12-31')

# 查看统计信息
stats = viz.get_statistics('300391')

# 绘制价格走势图
viz.plot_price_chart('300391', save_path='chart.png')

# 绘制K线图
viz.plot_candlestick('300391', days=60, save_path='kline.png')

# 多股票对比
viz.plot_multiple_stocks(['300391', '301200'], save_path='compare.png')

# 关闭连接
viz.close()
```

### 3. 运行示例脚本

```bash
python example_usage.py
```

## 参数说明

- `--code, -c`: 股票代码（必需，除非使用--list）
- `--list, -l`: 列出所有股票代码
- `--start, -s`: 开始日期 (YYYY-MM-DD)
- `--end, -e`: 结束日期 (YYYY-MM-DD)
- `--type, -t`: 图表类型 (price/candlestick/compare)
- `--days, -d`: 显示最近N天的数据（仅K线图）
- `--save`: 保存图表路径
- `--stats`: 显示统计信息

## 图表类型

### 1. 价格走势图 (price)
- 显示收盘价走势
- 显示最高-最低价区间
- 显示成交量

### 2. K线图 (candlestick)
- 传统蜡烛图
- 红色表示上涨，绿色表示下跌
- 显示开盘、收盘、最高、最低价

### 3. 对比图 (compare)
- 多股票收益率对比
- 以第一天的收盘价为基准进行归一化
- 方便比较不同股票的相对表现

## 注意事项

1. 确保数据库文件 `quant_database.db` 存在
2. 如果图表中文显示乱码，请安装中文字体
3. 图表保存路径可以是相对路径或绝对路径
4. 日期格式必须为 YYYY-MM-DD

## 示例输出

运行示例脚本会生成：
- `{股票代码}_price.png` - 价格走势图
- `{股票代码}_candlestick.png` - K线图
- `stock_compare.png` - 多股票对比图
