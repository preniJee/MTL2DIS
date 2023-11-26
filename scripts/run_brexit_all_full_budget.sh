#!/bin/bash
# run with 4 random seeds for each annotator with save hyperparameters

cd ..
source venv/bin/activate

GPUs=(3 2)

annotators_1=("Ann1,Ann2,Ann3,Ann4,Ann5,Ann6")
# Calculate the midpoint index for parallel execution
# midpoint=$((${#annotators[@]} / 2))

# Loop over the first half of tasks and run in parallel
for ann_index in  "${!annotators_1[@]}" ; do
    
    tasks_1=${annotators_1[$ann_index]}
    
    # Get current gpu
    gpu_index=$(($ann_index % ${#GPUs[@]}))
    gpu=${GPUs[$gpu_index]}
    
    echo "Running model for tasks: $tasks_1, gpu: $gpu"
    
    SESSION_NAME="${gpu}_${tasks_1}"
    
    screen -dmS "$SESSION_NAME" bash -c "CUDA_VISIBLE_DEVICES=$gpu python mtl_main.py --budget "4704" \
                                                --mtl_tasks "$tasks_1" \
                                                --n_fewshot_tasks 0 \
                                                --run_sweep \
                                                --seed "0" ;
                                    "
done

# Wait for the background jobs to finish
# wait



CUDA_VISIBLE_DEVICES=3 python mtl_main.py \
                --mtl_tasks "Ann1,Ann2,Ann3,Ann4,Ann5,Ann6" \
                --budget 4704 \
                --n_fewshot_tasks 0 \
                --run_sweep --seed "0" ;