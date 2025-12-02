if (!(Test-Path "log")) { New-Item -ItemType Directory -Path "log" | Out-Null }
if (!(Test-Path "figure")) { New-Item -ItemType Directory -Path "figure" | Out-Null }

python q1.py *>&1 | Tee-Object -FilePath "log/q1.log"

python q2.py *>&1 | Tee-Object -FilePath "log/q2.log"

python q3.py *>&1 | Tee-Object -FilePath "log/q3.log"

Write-Host "Well done!"
Pause
