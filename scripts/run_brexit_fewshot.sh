#!/bin/bash
# run with 4 random seeds for each annotator with save hyperparameters

cd ..
source venv/bin/activate

GPUs=(5 6)

annotators=("Ann1" "Ann2" "Ann3" "Ann4" "Ann5" "Ann6")
k_shots=(16 32 64 128)
n_mtls=(3 4 5)
# Calculate the midpoint index for parallel execution
# midpoint=$((${#annotators[@]} / 2))

# Loop over the first half of tasks and run in parallel
for k_index in "${!k_shots[@]}" ; do
    
    k_shot=${k_shots[$k_index]}
    
    for ann_index in  "${!annotators[@]}" ; do

        few_shot_task=${annotators[$ann_index]}
        
        for n_index in  "${!n_mtls[@]}" ; do
        
            n_mtl=${n_mtls[$n_index]}
            
            # Get current gpu
            gpu_index=$(($ann_index % ${#GPUs[@]}))
            gpu=${GPUs[$gpu_index]}
            
            echo "Running model for : $few_shot_task, on $n_mtl mtl tasks, with $k_shot shot on gpu: $gpu"
            
            SESSION_NAME="${gpu}_${k_shot}_${few_shot_task}_${n_mtl}"
            
            screen -dmS "$SESSION_NAME" bash -c "CUDA_VISIBLE_DEVICES=$gpu python fewshot_main.py \
                                                            --few_shot_task "$few_shot_task" \
                                                            --n_mtl_tasks $n_mtl \
                                                            --seed "0" \
                                                            --balance_ratio 0.5 \
                                                            --epochs 30 \
                                                            --k_shot $k_shot \
                                                            --few_shot_sample_strategy "mv" ;
                                                "
        done
        
    done
    
done


# CUDA_VISIBLE_DEVICES=3 python fewshot_main.py \
#                                                             --few_shot_task "Ann1" \
#                                                             --n_mtl_tasks 3 \
#                                                             --balance_ratio 0.5 \
#                                                             --seed "0" \
#                                                             --epochs 30 \
#                                                             --k_shot 32 \
#                                                             --few_shot_sample_strategy "mv" ;
                                                