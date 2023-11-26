#!/bin/bash
# run with 4 random seeds for each annotator with save hyperparameters

cd ..
source venv/bin/activate

GPUs=(3 4)

annotators_1=("Ann1,Ann2,Ann3,Ann4" "Ann1,Ann2,Ann3,Ann5" "Ann1,Ann2,Ann3,Ann6" "Ann1,Ann2,Ann4,Ann5" "Ann1,Ann2,Ann4,Ann6")
annotators_2=("Ann1,Ann2,Ann5,Ann6" "Ann1,Ann3,Ann4,Ann5" "Ann1,Ann3,Ann4,Ann6" "Ann1,Ann3,Ann5,Ann6" "Ann1,Ann4,Ann5,Ann6")
annotators_3=("Ann2,Ann3,Ann4,Ann5" "Ann2,Ann3,Ann4,Ann6" "Ann2,Ann3,Ann5,Ann6" "Ann2,Ann4,Ann5,Ann6" "Ann3,Ann4,Ann5,Ann6")

# Calculate the midpoint index for parallel execution
# midpoint=$((${#annotators[@]} / 2))

# Loop over the first half of tasks and run in parallel
for ann_index in  "${!annotators_1[@]}" ; do
    
    tasks_1=${annotators_1[$ann_index]}
    tasks_2=${annotators_2[$ann_index]}
    tasks_3=${annotators_3[$ann_index]}
    echo "heyy $tasks"
    # Get current gpu
    gpu_index=$(($ann_index % ${#GPUs[@]}))
    gpu=${GPUs[$gpu_index]}
    
    echo "Running model for tasks: $tasks_1, gpu: $gpu"
    
    SESSION_NAME="${gpu}_${tasks_1}"
    
    screen -dmS "$SESSION_NAME" bash -c "CUDA_VISIBLE_DEVICES=$gpu python mtl_main.py \
                                                    --mtl_tasks "$tasks_1" \
                                                    --n_fewshot_tasks 2 \
                                                    --run_sweep \
                                                    --seed "0" ;
                                        CUDA_VISIBLE_DEVICES=$gpu python mtl_main.py \
                                                    --mtl_tasks "$tasks_2" \
                                                    --n_fewshot_tasks 2 \
                                                    --run_sweep \
                                                    --seed "0" ;
                                        CUDA_VISIBLE_DEVICES=$gpu python mtl_main.py \
                                                    --mtl_tasks "$tasks_3" \
                                                    --n_fewshot_tasks 2 \
                                                    --run_sweep \
                                                    --seed "0" ; 
                                        "
done

# Wait for the background jobs to finish
# wait



# CUDA_VISIBLE_DEVICES=3 python mtl_main.py \
#                                                     --mtl_tasks "Ann1,Ann2,Ann3" \
#                                                     --run_sweep \
#                                                     --seed "0"