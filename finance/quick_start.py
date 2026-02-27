#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速开始脚本 - 一键查询和可视化
"""
from query_visualize import StockQueryVisualizer

def quick_start():
    """快速开始示例"""
    print("=" * 60)
    print("股票数据查询和可视化工具 - 快速开始")
    print("=" * 60)
    
    viz = StockQueryVisualizer()
    
    try:
        # 获取所有股票代码
        codes = viz.list_stocks()
        
        if not codes:
            print("数据库中没有股票数据，请先运行 init_database.py")
            return
        
        print(f"\n找到 {len(codes)} 只股票: {', '.join(codes)}")
        
        # 使用第一只股票作为示例
        stock_code = codes[0]
        print(f"\n使用股票 {stock_code} 作为示例...")
        
        # 显示统计信息
        print("\n" + "-" * 60)
        print("统计信息:")
        print("-" * 60)
        stats = viz.get_statistics(stock_code)
        if stats:
            for key, value in stats.items():
                print(f"  {key}: {value}")
        
        # 询问用户想要查看什么
        print("\n" + "-" * 60)
        print("请选择要查看的图表:")
        print("  1. 价格走势图")
        print("  2. K线图（最近60天）")
        print("  3. 多股票对比（如果有多个股票）")
        print("  4. 全部")
        print("-" * 60)
        
        choice = input("\n请输入选择 (1-4，直接回车默认选择1): ").strip()
        
        if choice == '':
            choice = '1'
        
        if choice in ['1', '4']:
            print(f"\n正在生成 {stock_code} 的价格走势图...")
            viz.plot_price_chart(stock_code, save_path=f'{stock_code}_price.png')
        
        if choice in ['2', '4']:
            print(f"\n正在生成 {stock_code} 的K线图...")
            viz.plot_candlestick(stock_code, days=60, save_path=f'{stock_code}_candlestick.png')
        
        if choice in ['3', '4'] and len(codes) >= 2:
            print(f"\n正在生成多股票对比图...")
            viz.plot_multiple_stocks(codes[:min(3, len(codes))], save_path='stock_compare.png')
        
        print("\n完成！图表已保存到当前目录。")
        
    except KeyboardInterrupt:
        print("\n\n用户取消操作")
    except Exception as e:
        print(f"\n错误: {e}")
    finally:
        viz.close()

if __name__ == "__main__":
    quick_start()
