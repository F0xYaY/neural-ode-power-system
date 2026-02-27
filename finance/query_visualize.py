#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票数据查询和可视化工具
支持从SQLite数据库查询股票数据并生成图表
"""
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import argparse
import sys

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class StockQueryVisualizer:
    def __init__(self, db_path='quant_database.db'):
        """初始化查询可视化工具"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
    
    def query_stock(self, stock_code, start_date=None, end_date=None):
        """
        查询股票数据
        
        参数:
            stock_code: 股票代码
            start_date: 开始日期 (格式: YYYY-MM-DD)
            end_date: 结束日期 (格式: YYYY-MM-DD)
        """
        query = "SELECT * FROM daily_price WHERE stock_code = ?"
        params = [stock_code]
        
        if start_date:
            query += " AND trade_date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND trade_date <= ?"
            params.append(end_date)
        
        query += " ORDER BY trade_date"
        
        df = pd.read_sql_query(query, self.conn, params=params)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.set_index('trade_date', inplace=True)
        
        return df
    
    def list_stocks(self):
        """列出数据库中所有股票代码"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT stock_code FROM daily_price ORDER BY stock_code")
        codes = [row[0] for row in cursor.fetchall()]
        return codes
    
    def plot_price_chart(self, stock_code, start_date=None, end_date=None, save_path=None):
        """
        绘制价格走势图
        
        参数:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            save_path: 保存路径（可选）
        """
        df = self.query_stock(stock_code, start_date, end_date)
        
        if df.empty:
            print(f"未找到股票 {stock_code} 的数据")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # 1. 价格走势图
        ax1 = axes[0]
        ax1.plot(df.index, df['close'], label='收盘价', linewidth=2, color='#1f77b4')
        ax1.fill_between(df.index, df['low'], df['high'], alpha=0.3, color='lightblue', label='最高-最低')
        ax1.set_ylabel('价格 (元)', fontsize=12)
        ax1.set_title(f'股票 {stock_code} 价格走势图', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 2. 成交量图
        ax2 = axes[1]
        ax2.bar(df.index, df['volume'], alpha=0.6, color='orange', label='成交量')
        ax2.set_ylabel('成交量', fontsize=12)
        ax2.set_xlabel('日期', fontsize=12)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # 格式化日期
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.show()
    
    def plot_candlestick(self, stock_code, start_date=None, end_date=None, days=60, save_path=None):
        """
        绘制K线图（蜡烛图）
        
        参数:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            days: 显示最近N天的数据
            save_path: 保存路径（可选）
        """
        df = self.query_stock(stock_code, start_date, end_date)
        
        if df.empty:
            print(f"未找到股票 {stock_code} 的数据")
            return
        
        # 如果指定了days，只显示最近N天
        if days and len(df) > days:
            df = df.tail(days)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 绘制K线
        for i, (date, row) in enumerate(df.iterrows()):
            open_price = row['open']
            close_price = row['close']
            high_price = row['high']
            low_price = row['low']
            
            # 判断涨跌
            color = 'red' if close_price >= open_price else 'green'
            
            # 绘制影线
            ax.plot([i, i], [low_price, high_price], color='black', linewidth=1)
            
            # 绘制实体
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            ax.bar(i, body_height, bottom=body_bottom, color=color, alpha=0.8, width=0.6)
        
        # 设置x轴标签
        step = max(1, len(df) // 10)
        ax.set_xticks(range(0, len(df), step))
        ax.set_xticklabels([df.index[i].strftime('%Y-%m-%d') for i in range(0, len(df), step)], rotation=45)
        
        ax.set_ylabel('价格 (元)', fontsize=12)
        ax.set_title(f'股票 {stock_code} K线图 (最近 {len(df)} 天)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.show()
    
    def plot_multiple_stocks(self, stock_codes, start_date=None, end_date=None, save_path=None):
        """
        绘制多只股票的对比图
        
        参数:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            save_path: 保存路径（可选）
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = plt.cm.tab10(range(len(stock_codes)))
        
        for i, code in enumerate(stock_codes):
            df = self.query_stock(code, start_date, end_date)
            if not df.empty:
                # 归一化价格（以第一天的收盘价为基准）
                base_price = df['close'].iloc[0]
                normalized_price = (df['close'] / base_price - 1) * 100
                ax.plot(df.index, normalized_price, label=f'{code}', linewidth=2, color=colors[i])
        
        ax.set_ylabel('收益率 (%)', fontsize=12)
        ax.set_xlabel('日期', fontsize=12)
        ax.set_title('多股票收益率对比图', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.show()
    
    def get_statistics(self, stock_code):
        """获取股票统计信息"""
        df = self.query_stock(stock_code)
        
        if df.empty:
            return None
        
        stats = {
            '股票代码': stock_code,
            '数据条数': len(df),
            '日期范围': f"{df.index.min().strftime('%Y-%m-%d')} 至 {df.index.max().strftime('%Y-%m-%d')}",
            '最高价': df['high'].max(),
            '最低价': df['low'].min(),
            '平均收盘价': df['close'].mean(),
            '当前收盘价': df['close'].iloc[-1],
            '总成交量': df['volume'].sum(),
            '平均成交量': df['volume'].mean(),
        }
        
        # 计算涨跌幅
        if len(df) > 1:
            first_close = df['close'].iloc[0]
            last_close = df['close'].iloc[-1]
            stats['期间涨跌幅'] = f"{((last_close / first_close - 1) * 100):.2f}%"
        
        return stats
    
    def close(self):
        """关闭数据库连接"""
        self.conn.close()


def main():
    parser = argparse.ArgumentParser(description='股票数据查询和可视化工具')
    parser.add_argument('--code', '-c', type=str, help='股票代码')
    parser.add_argument('--list', '-l', action='store_true', help='列出所有股票代码')
    parser.add_argument('--start', '-s', type=str, help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--type', '-t', type=str, choices=['price', 'candlestick', 'compare'], 
                       default='price', help='图表类型: price(价格走势), candlestick(K线), compare(对比)')
    parser.add_argument('--days', '-d', type=int, help='显示最近N天的数据（仅K线图）')
    parser.add_argument('--save', type=str, help='保存图表路径')
    parser.add_argument('--stats', action='store_true', help='显示统计信息')
    
    args = parser.parse_args()
    
    viz = StockQueryVisualizer()
    
    try:
        if args.list:
            codes = viz.list_stocks()
            print("数据库中的股票代码:")
            for code in codes:
                print(f"  {code}")
            return
        
        if not args.code:
            print("请指定股票代码，使用 --code 或 -c 参数")
            print("或使用 --list 查看所有股票代码")
            return
        
        # 显示统计信息
        if args.stats:
            stats = viz.get_statistics(args.code)
            if stats:
                print("\n股票统计信息:")
                print("=" * 50)
                for key, value in stats.items():
                    print(f"{key}: {value}")
                print("=" * 50)
        
        # 绘制图表
        if args.type == 'price':
            viz.plot_price_chart(args.code, args.start, args.end, args.save)
        elif args.type == 'candlestick':
            viz.plot_candlestick(args.code, args.start, args.end, args.days, args.save)
        elif args.type == 'compare':
            codes = args.code.split(',')
            viz.plot_multiple_stocks(codes, args.start, args.end, args.save)
    
    finally:
        viz.close()


if __name__ == "__main__":
    main()
