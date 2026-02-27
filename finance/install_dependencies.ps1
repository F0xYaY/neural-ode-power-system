#
 安装同花顺 MCP 服务器依赖
Write-Host "正在安装同花顺 MCP 服务器依赖..." -ForegroundColor Green

# 检查 Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python 版本: $pythonVersion" -ForegroundColor Cyan
} catch {
    Write-Host "错误: 未找到 Python，请先安装 Python 3.8+" -ForegroundColor Red
    exit 1
}

# 安装 akshare
Write-Host "`n正在安装 akshare..." -ForegroundColor Yellow
pip install akshare --upgrade

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ akshare 安装成功" -ForegroundColor Green
} else {
    Write-Host "✗ akshare 安装失败" -ForegroundColor Red
    exit 1
}

Write-Host "`n所有依赖安装完成！" -ForegroundColor Green
Write-Host "现在可以运行 MCP 服务器了。" -ForegroundColor Cyan
