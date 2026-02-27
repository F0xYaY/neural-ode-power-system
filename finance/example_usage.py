#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查询可视化工具使用示例
"""
from query_visualize import StockQueryVisualizer

def example_usage():
    """使用示例"""
    viz = StockQueryVisualizer()
    
    try:
        # 1. 列出所有股票
        print("=" * 60)
        print("1. 列出数据库中的所有股票")
        print("=" * 60)
        codes = viz.list_stocks()
        print(f"共有 {len(codes)} 只股票:")
        for code in codes:
            print(f"  - {code}")
        
        # 2. 查看股票统计信息
        print("\n" + "=" * 60)
        print("2. 查看股票统计信息")
        print("=" * 60)
        if codes:
            stats = viz.get_statistics(codes[0])
            if stats:
                for key, value in stats.items():
                    print(f"{key}: {value}")
        
        # 3. 绘制价格走势图
        print("\n" + "=" * 60)
        print("3. 绘制价格走势图")
        print("=" * 60)
        if codes:
            print(f"正在绘制 {codes[0]} 的价格走势图...")
            viz.plot_price_chart(codes[0], save_path=f'{codes[0]}_price.png')
        
        # 4. 绘制K线图
        print("\n" + "=" * 60)
        print("4. 绘制K线图（最近60天）")
        print("=" * 60)
        if codes:
            print(f"正在绘制 {codes[0]} 的K线图...")
            viz.plot_candlestick(codes[0], days=60, save_path=f'{codes[0]}_candlestick.png')
        
        # 5. 多股票对比
        if len(codes) >= 2:
            print("\n" + "=" * 60)
            print("5. 多股票收益率对比")
            print("=" * 60)
            print(f"正在对比 {codes[0]} 和 {codes[1]}...")
            viz.plot_multiple_stocks(codes[:2], save_path='stock_compare.png')
        
    finally:
        viz.close()

if __name__ == "__main__":
    example_usage()
