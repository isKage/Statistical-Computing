#!/bin/bash

mkdir -p log
mkdir -p figure

python3 -u q1.py | tee log/q1.log

python3 -u q2.py | tee log/q2.log

echo "Well done!"