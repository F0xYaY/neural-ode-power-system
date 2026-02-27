# 金融数据工具文件夹

本文件夹包含所有与 **akshare**、**同花顺** 等金融数据相关的文件和工具。

## 📦 包含内容

- **MCP 服务器**: 同花顺 MCP 服务器主程序
- **依赖管理**: requirements.txt 和安装脚本
- **测试脚本**: 各种测试和查询脚本
- **文档**: README 和使用说明

## 📁 文件夹结构

```
finance/
├── mcp_server.py              # 同花顺 MCP 服务器主程序
├── requirements.txt           # Python 依赖包列表
├── install_dependencies.ps1   # 自动安装依赖脚本
├── README.md                  # 项目说明文档
├── 使用说明.md                # 使用指南
├── test_mcp.py               # MCP 服务器测试脚本
├── test_etf.py               # ETF 功能测试脚本
└── ...                        # 其他测试和查询脚本
```

## 🚀 快速开始

### 1. 安装依赖

```powershell
.\install_dependencies.ps1
```

或手动安装：

```bash
pip install -r requirements.txt
```

### 2. 测试 MCP 服务器

```bash
python test_mcp.py
```

### 3. 在 Cursor 中使用

MCP 服务器已配置在 `.cursor/mcp.json` 中，重启 Cursor 后即可使用。

## 📊 功能特性

- ✅ **股票实时行情查询**
- ✅ **K线数据获取**（日线、周线、月线）
- ✅ **ETF基金查询**
- ✅ **市场指数查询**
- ✅ **股票搜索功能**

## 🔧 数据源

- **akshare**: 免费开源金融数据接口
- **数据来源**: 同花顺、东方财富等
- **无需 API Key**: 直接使用

## 📝 使用示例

### 在 Cursor AI 聊天中：

```
查询平安银行（000001）的实时行情
```

```
获取159919（沪深300ETF）的K线图，最近30天
```

```
搜索包含"银行"的股票
```

## ⚠️ 注意事项

- 数据仅供学习研究使用，投资需谨慎
- akshare 数据可能有延迟，非实时数据
- 建议在交易时间外测试，避免频繁请求

## 📚 相关文档

- [akshare 官方文档](https://akshare.akfamily.cn/)
- [MCP 协议文档](https://modelcontextprotocol.io/)
