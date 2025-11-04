if not exist log mkdir log
if not exist figure mkdir figure

python q1_c.py > log\q1_c.log 2>&1
type log\q1_c.log

python q1_d.py > log\q1_d.log 2>&1
type log\q1_d.log

python q1_e.py > log\q1_e.log 2>&1
type log\q1_e.log

python q1_f.py > log\q1_f.log 2>&1
type log\q1_f.log


echo "Well done!"
pause
