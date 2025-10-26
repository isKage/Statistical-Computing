if not exist log mkdir log
if not exist figure mkdir figure

python q1.py > log\q1.log 2>&1
type log\q1.log

python q2.py > log\q2.log 2>&1
type log\q2.log

python q3_a.py > log\q3_a.log 2>&1
type log\q3_a.log

python q3_b.py > log\q3_b.log 2>&1
type log\q3_b.log

python q4_bc.py > log\q4_bc.log 2>&1
type log\q4_bc.log

python q4_d.py > log\q4_d.log 2>&1
type log\q4_d.log

python q4_e.py > log\q4_e.log 2>&1
type log\q4_e.log

python q4_fgh.py > log\q4_fgh.log 2>&1
type log\q4_fgh.log

echo "Well done!"
pause
