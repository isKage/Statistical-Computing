if (!(Test-Path "log")) { New-Item -ItemType Directory -Path "log" | Out-Null }
if (!(Test-Path "figure")) { New-Item -ItemType Directory -Path "figure" | Out-Null }

python q1.py *>&1 | Tee-Object -FilePath "log/q1.log"

python q2.py *>&1 | Tee-Object -FilePath "log/q2.log"

python q3_a.py *>&1 | Tee-Object -FilePath "log/q3_a.log"
python q3_b.py *>&1 | Tee-Object -FilePath "log/q3_b.log"

python q4_bc.py *>&1 | Tee-Object -FilePath "log/q4_bc.log"
python q4_d.py *>&1 | Tee-Object -FilePath "log/q4_d.log"
python q4_e.py *>&1 | Tee-Object -FilePath "log/q4_e.log"
python q4_fgh.py *>&1 | Tee-Object -FilePath "log/q4_fgh.log"

Write-Host "Well done!"
Pause
