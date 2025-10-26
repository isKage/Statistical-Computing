#!/bin/bash

mkdir -p log
mkdir -p figure

python3 q1.py | tee log/q1.log

python3 q2.py | tee log/q2.log

python3 q3_a.py | tee log/q3_a.log
python3 q3_b.py | tee log/q3_b.log

python3 q4_bc.py | tee log/q4_bc.log
python3 q4_d.py | tee log/q4_d.log
python3 q4_e.py | tee log/q4_e.log
python3 q4_fgh.py | tee log/q4_fgh.log

echo "Well done!"