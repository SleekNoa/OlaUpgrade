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

Test-OlaChat "General Test" "What is the capital of France?"
