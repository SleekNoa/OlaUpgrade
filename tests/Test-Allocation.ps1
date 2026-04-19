$uri = "http://localhost:8000/v1/chat/completions"
$headers = @{ "Content-Type" = "application/json" }

function Test-OlaChat {
    param([string]$TestName, [string]$Prompt)

    Write-Host "`n=== $TestName ===" -ForegroundColor Cyan
    $body = @{
        messages = @(
            @{ role = "user"; content = $Prompt }
        )
        model = "ola-agent"
    } | ConvertTo-Json -Depth 10

    Invoke-RestMethod -Uri $uri -Method Post -Body $body -Headers $headers |
        Select-Object -ExpandProperty choices |
        Select-Object -ExpandProperty message |
        Select-Object content
}

$prompt = @'
Allocate these tasks:
T1: Database migration (requires db)
T2: Network configuration (requires net)
T3: Frontend redesign (requires ui)

To these agents:
A1: skills [db, backend], capacity 2
A2: skills [net, infra], capacity 1
A3: skills [ui, design], capacity 3
'@

Test-OlaChat "Allocation Test" $prompt
