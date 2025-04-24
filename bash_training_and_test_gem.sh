#!/bin/bash

#training and test CL...
echo " Running gem EXP"
echo $(pwd)


# experiment for gem


# python train_CL.py  --model gem --buffer_size 1000  --n_epochs 1 --alpha 1   --beta 0.5  --experiment_index 1

# python test_CL.py   --model gem --buffer_size 1000  --num_tasks 8   --experiment_index 1

# python train_CL.py  --model gem --buffer_size 1000  --n_epochs 1 --alpha 1   --beta 0.5  --restart_training True --restart_pre_task_num 6 --experiment_index 2
# python test_CL.py   --model gem --buffer_size 1000  --num_tasks 8   --experiment_index 2



python train_CL.py  --model agem --buffer_size 500  --n_epochs 1 --alpha 1   --beta 0.5  --experiment_index 3
python test_CL.py   --model agem --buffer_size 500  --num_tasks 8   --experiment_index 3

python train_CL.py  --model agem --buffer_size 1000  --n_epochs 1 --alpha 1   --beta 0.5  --experiment_index 4
python test_CL.py   --model agem --buffer_size 1000  --num_tasks 8   --experiment_index 4

python train_CL.py  --model agem --buffer_size 2000  --n_epochs 1 --alpha 1   --beta 0.5  --experiment_index 5
python test_CL.py   --model agem --buffer_size 2000  --num_tasks 8   --experiment_index 5

python train_CL.py  --model agem --buffer_size 4000  --n_epochs 1 --alpha 1   --beta 0.5  --experiment_index 6
python test_CL.py   --model agem --buffer_size 4000  --num_tasks 8   --experiment_index 6
echo "Ours Finished"