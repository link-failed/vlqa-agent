# PowerShell 脚本 - 顺序运行多个命令并打印进度

# 定义命令列表
$commands = @(
    'python baseline/run2.py  --model-id "gpt-4o" --use-azure-auth --max-tasks 450 --split default --concurrency 8'
)

# 遍历执行
for ($i = 0; $i -lt $commands.Count; $i++) {
    Write-Host "=========================="
    Write-Host "开始执行第 $($i + 1) 个任务，共 $($commands.Count) 个"
    Write-Host "命令: $($commands[$i])"
    Write-Host "=========================="

    # 执行命令
    Invoke-Expression $commands[$i]

    Write-Host "第 $($i + 1) 个任务执行完成"
    Write-Host ""
}
