#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
同花顺 MCP 服务器 - 集成真实数据接口
使用 akshare 库获取股票数据（免费，无需 API key）
"""
import asyncio
import json
import sys
from typing import Any, Dict, List, Optional
from datetime import datetime
import traceback

# 尝试导入 akshare，如果未安装则使用示例数据
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    print("警告: akshare 未安装，将使用示例数据。安装命令: pip install akshare", file=sys.stderr)

# MCP 协议基础实现
class TonghuashunMCPServer:
    def __init__(self):
        self.tools = {
            "get_stock_quote": {
                "name": "get_stock_quote",
                "description": "获取股票实时行情",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "股票代码，如 '000001' (深交所) 或 '600000' (上交所)"
                        }
                    },
                    "required": ["code"]
                }
            },
            "get_kline_data": {
                "name": "get_kline_data",
                "description": "获取K线数据",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "股票代码"
                        },
                        "period": {
                            "type": "string",
                            "description": "周期：'daily' (日线), 'weekly' (周线), 'monthly' (月线)",
                            "default": "daily"
                        },
                        "count": {
                            "type": "integer",
                            "description": "获取数量",
                            "default": 100
                        }
                    },
                    "required": ["code"]
                }
            },
            "search_stock": {
                "name": "search_stock",
                "description": "搜索股票",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "keyword": {
                            "type": "string",
                            "description": "搜索关键词（股票代码或名称）"
                        }
                    },
                    "required": ["keyword"]
                }
            },
            "get_market_index": {
                "name": "get_market_index",
                "description": "获取市场指数",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "index_code": {
                            "type": "string",
                            "description": "指数代码，如 'sh000001' (上证指数), 'sz399001' (深证成指), 'sz399006' (创业板指)",
                            "default": "sh000001"
                        }
                    }
                }
            },
            "search_etf": {
                "name": "search_etf",
                "description": "搜索ETF基金",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "keyword": {
                            "type": "string",
                            "description": "搜索关键词（ETF代码或名称，如 '510050' 或 '上证50'）"
                        }
                    },
                    "required": ["keyword"]
                }
            },
            "get_etf_kline": {
                "name": "get_etf_kline",
                "description": "获取ETF基金K线数据",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "ETF基金代码，如 '510050' (华夏上证50ETF), '159919' (沪深300ETF)"
                        },
                        "period": {
                            "type": "string",
                            "description": "周期：'daily' (日线), 'weekly' (周线), 'monthly' (月线)",
                            "default": "daily"
                        },
                        "count": {
                            "type": "integer",
                            "description": "获取数量",
                            "default": 100
                        }
                    },
                    "required": ["code"]
                }
            }
        }
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理 MCP 请求"""
        method = request.get("method")
        params = request.get("params", {})
        
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "tonghuashun-mcp",
                        "version": "1.0.0",
                        "dataSource": "akshare" if AKSHARE_AVAILABLE else "demo"
                    }
                }
            }
        
        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": {
                    "tools": list(self.tools.values())
                }
            }
        
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            result = await self.execute_tool(tool_name, arguments)
            
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, ensure_ascii=False, indent=2)
                        }
                    ]
                }
            }
        
        else:
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """执行工具函数"""
        try:
            if tool_name == "get_stock_quote":
                return await self._get_stock_quote(arguments.get("code"))
            elif tool_name == "get_kline_data":
                return await self._get_kline_data(
                    arguments.get("code"),
                    arguments.get("period", "daily"),
                    arguments.get("count", 100)
                )
            elif tool_name == "search_stock":
                return await self._search_stock(arguments.get("keyword"))
            elif tool_name == "get_market_index":
                return await self._get_market_index(arguments.get("index_code", "sh000001"))
            elif tool_name == "search_etf":
                return await self._search_etf(arguments.get("keyword"))
            elif tool_name == "get_etf_kline":
                return await self._get_etf_kline(
                    arguments.get("code"),
                    arguments.get("period", "daily"),
                    arguments.get("count", 100)
                )
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            return {
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _normalize_stock_code(self, code: str) -> str:
        """标准化股票代码格式"""
        code = code.strip()
        # 如果代码以 sh/sz 开头，保持不变
        if code.startswith(('sh', 'sz')):
            return code
        # 否则根据代码判断市场
        if code.startswith(('60', '68', '51')):  # 上交所（包括ETF 51xxxx）
            return f"sh{code}"
        elif code.startswith(('00', '30', '159')):  # 深交所（包括ETF 159xxx）
            return f"sz{code}"
        else:
            # 默认尝试深交所
            return f"sz{code}"
    
    async def _get_stock_quote(self, code: str) -> Dict[str, Any]:
        """获取股票实时行情"""
        if not AKSHARE_AVAILABLE:
            return {
                "code": code,
                "name": "示例股票",
                "price": 10.50,
                "change": 0.25,
                "change_percent": 2.44,
                "volume": 1000000,
                "amount": 10500000,
                "high": 10.60,
                "low": 10.30,
                "open": 10.35,
                "pre_close": 10.25,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "note": "akshare 未安装，返回示例数据。安装: pip install akshare"
            }
        
        try:
            # 标准化代码
            normalized_code = self._normalize_stock_code(code)
            
            # 使用 akshare 获取实时行情
            # 注意：akshare 的实时行情接口可能需要根据最新版本调整
            df = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ak.stock_zh_a_spot_em()
            )
            
            # 查找指定股票
            stock_code = code if len(code) == 6 else code[-6:]
            stock_data = df[df['代码'] == stock_code]
            
            if stock_data.empty:
                return {
                    "error": f"未找到股票代码: {code}",
                    "suggestion": "请检查股票代码是否正确"
                }
            
            row = stock_data.iloc[0]
            
            current_price = float(row['最新价'])
            change = float(row['涨跌额'])
            change_percent = float(row['涨跌幅'])
            pre_close = current_price - change
            
            return {
                "code": code,
                "name": str(row['名称']),
                "price": current_price,
                "change": change,
                "change_percent": change_percent,
                "volume": int(row['成交量']),
                "amount": float(row['成交额']),
                "high": float(row['最高']),
                "low": float(row['最低']),
                "open": float(row['今开']),
                "pre_close": pre_close,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataSource": "akshare"
            }
        except Exception as e:
            return {
                "error": str(e),
                "code": code,
                "traceback": traceback.format_exc()
            }
    
    async def _get_kline_data(self, code: str, period: str, count: int) -> Dict[str, Any]:
        """获取K线数据"""
        if not AKSHARE_AVAILABLE:
            return {
                "code": code,
                "period": period,
                "count": count,
                "data": [],
                "message": "akshare 未安装，无法获取真实数据。安装: pip install akshare"
            }
        
        try:
            normalized_code = self._normalize_stock_code(code)
            
            # 根据周期选择不同的接口
            # akshare的period参数使用英文：'daily', 'weekly', 'monthly'
            period_map = {
                "daily": "daily",
                "weekly": "weekly", 
                "monthly": "monthly"
            }
            
            period_str = period_map.get(period, "daily")
            
            df = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ak.stock_zh_a_hist(symbol=normalized_code, period=period_str, adjust="qfq")
            )
            
            # 限制返回数量
            df = df.tail(count)
            
            # 转换为字典列表
            data = []
            for _, row in df.iterrows():
                data.append({
                    "date": str(row['日期']),
                    "open": float(row['开盘']),
                    "high": float(row['最高']),
                    "low": float(row['最低']),
                    "close": float(row['收盘']),
                    "volume": int(row['成交量']),
                    "amount": float(row['成交额']) if '成交额' in row else None
                })
            
            return {
                "code": code,
                "period": period,
                "count": len(data),
                "data": data,
                "dataSource": "akshare"
            }
        except Exception as e:
            return {
                "error": str(e),
                "code": code,
                "period": period,
                "traceback": traceback.format_exc()
            }
    
    async def _search_stock(self, keyword: str) -> Dict[str, Any]:
        """搜索股票"""
        if not AKSHARE_AVAILABLE:
            return {
                "keyword": keyword,
                "results": [],
                "message": "akshare 未安装，无法搜索。安装: pip install akshare"
            }
        
        try:
            # 获取所有股票列表
            df = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ak.stock_info_a_code_name()
            )
            
            # 搜索匹配的股票
            keyword_lower = keyword.lower()
            results = []
            
            for _, row in df.iterrows():
                code = str(row['code'])
                name = str(row['name'])
                
                if keyword_lower in code.lower() or keyword_lower in name.lower():
                    results.append({
                        "code": code,
                        "name": name
                    })
                    if len(results) >= 20:  # 限制返回数量
                        break
            
            return {
                "keyword": keyword,
                "count": len(results),
                "results": results,
                "dataSource": "akshare"
            }
        except Exception as e:
            return {
                "error": str(e),
                "keyword": keyword,
                "traceback": traceback.format_exc()
            }
    
    async def _get_market_index(self, index_code: str) -> Dict[str, Any]:
        """获取市场指数"""
        if not AKSHARE_AVAILABLE:
            index_names = {
                "sh000001": "上证指数",
                "sz399001": "深证成指",
                "sz399006": "创业板指"
            }
            return {
                "code": index_code,
                "name": index_names.get(index_code, "未知指数"),
                "value": 3000.00,
                "change": 15.50,
                "change_percent": 0.52,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "note": "akshare 未安装，返回示例数据"
            }
        
        try:
            # 获取指数实时行情
            df = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ak.index_zh_a_hist(symbol=index_code, period="日k", start_date="20240101", end_date="")
            )
            
            if df.empty:
                return {"error": f"未找到指数: {index_code}"}
            
            # 获取最新数据
            latest = df.iloc[-1]
            
            # 计算涨跌
            current_value = float(latest['收盘'])
            if len(df) > 1:
                prev_value = float(df.iloc[-2]['收盘'])
                change = current_value - prev_value
                change_percent = (change / prev_value) * 100
            else:
                change = 0
                change_percent = 0
            
            index_names = {
                "sh000001": "上证指数",
                "sz399001": "深证成指",
                "sz399006": "创业板指"
            }
            
            return {
                "code": index_code,
                "name": index_names.get(index_code, "未知指数"),
                "value": current_value,
                "change": change,
                "change_percent": change_percent,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataSource": "akshare"
            }
        except Exception as e:
            return {
                "error": str(e),
                "code": index_code,
                "traceback": traceback.format_exc()
            }
    
    async def _search_etf(self, keyword: str) -> Dict[str, Any]:
        """搜索ETF基金"""
        if not AKSHARE_AVAILABLE:
            return {
                "keyword": keyword,
                "results": [],
                "message": "akshare 未安装，无法搜索。安装: pip install akshare"
            }
        
        try:
            # 使用股票列表，筛选ETF代码（51开头或159开头）
            df_all = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ak.stock_info_a_code_name()
            )
            
            # ETF代码特征：上交所51xxxx，深交所159xxx
            etf_codes = df_all[
                df_all['code'].str.startswith('51') | 
                df_all['code'].str.startswith('159')
            ]
            
            # 搜索匹配的ETF
            keyword_lower = keyword.lower()
            results = []
            
            for _, row in etf_codes.iterrows():
                code = str(row['code'])
                name = str(row['name'])
                
                # 检查代码或名称是否匹配
                if keyword_lower in code.lower() or keyword_lower in name.lower():
                    results.append({
                        "code": code,
                        "name": name
                    })
                    if len(results) >= 20:  # 限制返回数量
                        break
            
            return {
                "keyword": keyword,
                "count": len(results),
                "results": results,
                "dataSource": "akshare"
            }
        except Exception as e:
            return {
                "error": str(e),
                "keyword": keyword,
                "traceback": traceback.format_exc()
            }
    
    async def _get_etf_kline(self, code: str, period: str, count: int) -> Dict[str, Any]:
        """获取ETF基金K线数据"""
        if not AKSHARE_AVAILABLE:
            return {
                "code": code,
                "period": period,
                "count": count,
                "data": [],
                "message": "akshare 未安装，无法获取真实数据。安装: pip install akshare"
            }
        
        try:
            # ETF基金在交易所也是作为股票交易的，可以使用股票接口
            # 标准化代码格式
            normalized_code = self._normalize_stock_code(code)
            
            # 根据周期选择不同的接口（与股票相同）
            # akshare的period参数使用英文：'daily', 'weekly', 'monthly'
            period_map = {
                "daily": "daily",
                "weekly": "weekly", 
                "monthly": "monthly"
            }
            
            period_str = period_map.get(period, "daily")
            
            df = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ak.stock_zh_a_hist(symbol=normalized_code, period=period_str, adjust="qfq")
            )
            
            # 检查数据是否为空
            if df is None or (hasattr(df, 'empty') and df.empty):
                # 尝试不带前缀的代码
                if not code.startswith(('sh', 'sz')):
                    try:
                        df = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: ak.stock_zh_a_hist(symbol=code, period=period_str, adjust="qfq")
                        )
                    except:
                        pass
                
                if df is None or (hasattr(df, 'empty') and df.empty):
                    return {
                        "error": f"未找到ETF基金代码: {code}",
                        "suggestion": "请检查ETF代码是否正确（上交所ETF: 51xxxx, 深交所ETF: 159xxx）",
                        "tried_code": normalized_code
                    }
            
            # 限制返回数量
            df = df.tail(count)
            
            # 获取ETF名称（从实时行情中）
            etf_name = f"ETF基金 {code}"
            try:
                spot_df = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: ak.stock_zh_a_spot_em()
                )
                stock_code = code if len(code) == 6 else code[-6:]
                name_data = spot_df[spot_df['代码'] == stock_code]
                if not name_data.empty:
                    etf_name = str(name_data.iloc[0]['名称'])
            except:
                pass  # 如果获取名称失败，使用默认名称
            
            # 转换为字典列表
            data = []
            for _, row in df.iterrows():
                data.append({
                    "date": str(row['日期']),
                    "open": float(row['开盘']),
                    "high": float(row['最高']),
                    "low": float(row['最低']),
                    "close": float(row['收盘']),
                    "volume": int(row['成交量']),
                    "amount": float(row['成交额']) if '成交额' in row else None
                })
            
            return {
                "code": code,
                "name": etf_name,
                "period": period,
                "count": len(data),
                "data": data,
                "dataSource": "akshare"
            }
        except Exception as e:
            return {
                "error": str(e),
                "code": code,
                "period": period,
                "traceback": traceback.format_exc()
            }


async def main():
    """主函数 - 运行 MCP 服务器"""
    server = TonghuashunMCPServer()
    
    # 从 stdin 读取请求，向 stdout 写入响应
    while True:
        try:
            line = await asyncio.get_event_loop().run_in_executor(
                None, sys.stdin.readline
            )
            if not line:
                break
            
            request = json.loads(line.strip())
            response = await server.handle_request(request)
            
            print(json.dumps(response, ensure_ascii=False))
            sys.stdout.flush()
        
        except json.JSONDecodeError:
            continue
        except Exception as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
            print(json.dumps(error_response, ensure_ascii=False))
            sys.stdout.flush()


if __name__ == "__main__":
    asyncio.run(main())
