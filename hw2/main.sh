#!/bin/bash

mkdir -p log
mkdir -p figure

python3 -u q1_c.py | tee log/q1_c.log

python3 -u q1_d.py | tee log/q1_d.log

python3 -u q1_e.py | tee log/q1_e.log

python3 -u q1_f.py | tee log/q1_f.log

echo "Well done!"