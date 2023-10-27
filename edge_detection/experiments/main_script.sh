#!/bin/bash
seeds=(1 2 3 4 5 6 7 8 9 10)
n_samples=(15 25 35 45)

for s in ${seeds[@]}; do
  for n in ${n_samples[@]}; do
    echo "python main.py --n ${n} --d 30 --epochs 5001 --seed ${s}"
	done
done