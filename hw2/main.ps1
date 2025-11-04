if (!(Test-Path "log")) { New-Item -ItemType Directory -Path "log" | Out-Null }
if (!(Test-Path "figure")) { New-Item -ItemType Directory -Path "figure" | Out-Null }

python q1_c.py *>&1 | Tee-Object -FilePath "log/q1_c.log"

python q1_d.py *>&1 | Tee-Object -FilePath "log/q1_d.log"

python q1_e.py *>&1 | Tee-Object -FilePath "log/q1_e.log"

python q1_f.py *>&1 | Tee-Object -FilePath "log/q1_f.log"

Write-Host "Well done!"
Pause
