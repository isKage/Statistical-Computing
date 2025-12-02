if not exist log mkdir log
if not exist figure mkdir figure

python q1.py > log\q1.log 2>&1
type log\q1.log

python q2.py > log\q2.log 2>&1
type log\q2.log

python q3.py > log\q3.log 2>&1
type log\q3.log

echo "Well done!"
pause
