#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速运行市场动力学模型
"""
from market_dynamics_model import compare_market_scenarios, parameter_sensitivity_analysis

if __name__ == "__main__":
    print("=" * 60)
    print("市场动力学模型 - 快速运行")
    print("=" * 60)
    
    # 运行对比实验
    print("\n1. 运行市场对比实验...")
    compare_market_scenarios(days=200, save_path='market_dynamics_comparison.png')
    
    # 运行参数敏感性分析
    print("\n2. 运行参数敏感性分析...")
    parameter_sensitivity_analysis()
    
    print("\n完成！所有图表已生成。")
