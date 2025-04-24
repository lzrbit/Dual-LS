#!/bin/bash

#training and test CL...
echo " Running Continual Training EXP"
echo $(pwd)


experiment 1-9    --stable_model_update_freq 0.70  --plastic_model_update_freq 0.90

python train_CL.py  --model clser --buffer_size 1000  --n_epochs 1 --alpha 1   --beta 0.5 --experiment_index 1
python test_CL.py   --model clser --buffer_size 1000  --num_tasks 8   --experiment_index 1

python train_CL.py  --model clser --buffer_size 1000  --n_epochs 1 --alpha 0.5 --beta 1 --experiment_index 2
python test_CL.py   --model clser --buffer_size 1000  --num_tasks 8 --experiment_index 2


python train_CL.py  --model clser --buffer_size 1000  --n_epochs 1 --alpha 1   --beta 1 --experiment_index 3
python test_CL.py   --model clser --buffer_size 1000  --num_tasks 8   --experiment_index 3


python train_CL.py  --model clser --buffer_size 1000  --n_epochs 2 --alpha 1   --beta 0.5 --experiment_index 4
python test_CL.py   --model clser --buffer_size 1000  --num_tasks 8   --experiment_index 4


python train_CL.py  --model clser --buffer_size 1000  --n_epochs 2 --alpha 0.5 --beta 1 --experiment_index 5
python test_CL.py   --model clser --buffer_size 1000  --num_tasks 8    --experiment_index 5


python train_CL.py  --model clser --buffer_size 1000  --n_epochs 2 --alpha 1   --beta 1 --experiment_index 6
python test_CL.py   --model clser --buffer_size 1000  --num_tasks 8     --experiment_index 6


python train_CL.py  --model clser --buffer_size 1000  --n_epochs 3 --alpha 1   --beta 0.5 --experiment_index 7
python test_CL.py   --model clser --buffer_size 1000  --num_tasks 8   --experiment_index 7


python train_CL.py  --model clser --buffer_size 1000  --n_epochs 3 --alpha 0.5 --beta 1 --experiment_index 8
python test_CL.py   --model clser --buffer_size 1000  --num_tasks 8    --experiment_index 8


python train_CL.py  --model clser --buffer_size 1000  --n_epochs 3 --alpha 1   --beta 1 --experiment_index 9
python test_CL.py   --model clser --buffer_size 1000  --num_tasks 8    --experiment_index 9


experiment 10-18    --stable_model_update_freq 0.50  --plastic_model_update_freq 0.95

python train_CL.py  --model clser --buffer_size 1000  --n_epochs 1 --alpha 1   --beta 0.5 --experiment_index 10
python test_CL.py   --model clser --buffer_size 1000  --num_tasks 8   --experiment_index 10

python train_CL.py  --model clser --buffer_size 1000  --n_epochs 1 --alpha 0.5 --beta 1 --experiment_index 11
python test_CL.py   --model clser --buffer_size 1000  --num_tasks 8 --experiment_index 11


python train_CL.py  --model clser --buffer_size 1000  --n_epochs 1 --alpha 1   --beta 1 --experiment_index 12
python test_CL.py   --model clser --buffer_size 1000  --num_tasks 8   --experiment_index 12


python train_CL.py  --model clser --buffer_size 1000  --n_epochs 2 --alpha 1   --beta 0.5 --experiment_index 13
python test_CL.py   --model clser --buffer_size 1000  --num_tasks 8   --experiment_index 13


python train_CL.py  --model clser --buffer_size 1000  --n_epochs 2 --alpha 0.5 --beta 1 --experiment_index 14
python test_CL.py   --model clser --buffer_size 1000  --num_tasks 8    --experiment_index 14


python train_CL.py  --model clser --buffer_size 1000  --n_epochs 2 --alpha 1   --beta 1 --experiment_index 15
python test_CL.py   --model clser --buffer_size 1000  --num_tasks 8     --experiment_index 15


python train_CL.py  --model clser --buffer_size 1000  --n_epochs 3 --alpha 1   --beta 0.5 --experiment_index 16
python test_CL.py   --model clser --buffer_size 1000  --num_tasks 8   --experiment_index 16


python train_CL.py  --model clser --buffer_size 1000  --n_epochs 3 --alpha 0.5 --beta 1 --experiment_index 17
python test_CL.py   --model clser --buffer_size 1000  --num_tasks 8    --experiment_index 17


python train_CL.py  --model clser --buffer_size 1000  --n_epochs 3 --alpha 1   --beta 1 --experiment_index 18
python test_CL.py   --model clser --buffer_size 1000  --num_tasks 8    --experiment_index 18






experiment 19-27    --stable_model_update_freq 0.75  --plastic_model_update_freq 0.85

python train_CL.py  --model clser --buffer_size 1000  --n_epochs 1 --alpha 1   --beta 0.5 --experiment_index 19
python test_CL.py   --model clser --buffer_size 1000  --num_tasks 8   --experiment_index 19

python train_CL.py  --model clser --buffer_size 1000  --n_epochs 1 --alpha 0.5 --beta 1 --experiment_index 20
python test_CL.py   --model clser --buffer_size 1000  --num_tasks 8 --experiment_index 20


python train_CL.py  --model clser --buffer_size 1000  --n_epochs 1 --alpha 1   --beta 1 --experiment_index 21
python test_CL.py   --model clser --buffer_size 1000  --num_tasks 8   --experiment_index 21


python train_CL.py  --model clser --buffer_size 1000  --n_epochs 2 --alpha 1   --beta 0.5 --experiment_index 22
python test_CL.py   --model clser --buffer_size 1000  --num_tasks 8   --experiment_index 22


python train_CL.py  --model clser --buffer_size 1000  --n_epochs 2 --alpha 0.5 --beta 1 --experiment_index 23
python test_CL.py   --model clser --buffer_size 1000  --num_tasks 8    --experiment_index 23


python train_CL.py  --model clser --buffer_size 1000  --n_epochs 2 --alpha 1   --beta 1 --experiment_index 24
python test_CL.py   --model clser --buffer_size 1000  --num_tasks 8     --experiment_index 24


python train_CL.py  --model clser --buffer_size 1000  --n_epochs 3 --alpha 1   --beta 0.5 --experiment_index 25
python test_CL.py   --model clser --buffer_size 1000  --num_tasks 8   --experiment_index 25


python train_CL.py  --model clser --buffer_size 1000  --n_epochs 3 --alpha 0.5 --beta 1 --experiment_index 26
python test_CL.py   --model clser --buffer_size 1000  --num_tasks 8    --experiment_index 26


python train_CL.py  --model clser --buffer_size 1000  --n_epochs 3 --alpha 1   --beta 1 --experiment_index 27
python test_CL.py   --model clser --buffer_size 1000  --num_tasks 8    --experiment_index 27



echo "Ours Finished"