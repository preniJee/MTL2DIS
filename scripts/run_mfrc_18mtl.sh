#!/bin/bash
# run with 4 random seeds for each annotator with save hyperparameters

cd ..
source venv/bin/activate

GPUs=(2 4 5 6)

annotators_1=('2,3,4,5,6,7,8,9,12,14,15,16,17,18,19,20,22,23'
 '0,1,2,3,5,6,7,8,9,11,12,13,15,17,18,20,22,23'
 '0,2,3,4,5,6,8,9,10,11,12,13,14,15,17,18,20,23'
 '0,2,3,4,5,6,7,8,9,11,13,14,17,18,19,20,21,22'
 )


annotators_2=('0,1,2,3,5,6,8,10,11,14,15,17,18,19,20,21,22,23'
 '0,2,3,5,6,7,8,9,12,14,15,17,18,19,20,21,22,23'
 '0,1,2,5,7,9,10,11,12,14,15,16,17,18,19,20,21,22'
 '0,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,21')

 annotators_3=('1,2,4,5,6,7,8,9,10,11,12,13,16,18,19,21,22,23'
 '0,1,2,3,4,6,8,10,11,12,13,15,16,17,18,19,21,22'
 '0,5,6,8,9,10,11,12,13,14,15,16,17,18,19,21,22,23'
 '0,2,3,4,5,9,10,11,12,13,14,17,18,19,20,21,22,23'
 )

 annotators_4=('0,1,3,4,6,7,8,9,10,12,14,15,16,18,20,21,22,23'
 '0,2,3,4,5,6,7,8,9,11,12,13,17,19,20,21,22,23'
 '0,1,2,4,6,7,8,9,11,12,15,16,17,18,19,20,21,23'
 '0,1,2,4,6,9,10,11,12,13,16,17,18,19,20,21,22,23'
 )

 annotators_5=('0,1,2,4,6,7,9,10,12,13,15,16,17,18,19,20,21,23'
 '0,1,2,3,4,5,6,8,9,10,11,13,15,17,19,20,22,23'
 '0,1,3,5,6,9,10,11,12,13,14,15,16,17,18,19,22,23'
 '1,3,4,6,8,9,10,11,12,13,14,15,16,17,18,19,21,22')



# Loop over the first half of tasks and run in parallel
for ann_index in  "${!annotators_1[@]}" ; do
    
    tasks_1=${annotators_1[$ann_index]}
    tasks_2=${annotators_2[$ann_index]}
    tasks_3=${annotators_3[$ann_index]}
    tasks_4=${annotators_4[$ann_index]}
    tasks_5=${annotators_5[$ann_index]}

    # Get current gpu
    gpu_index=$(($ann_index % ${#GPUs[@]}))
    gpu=${GPUs[$gpu_index]}
    
    echo "Running model for tasks: $tasks_1, gpu: $gpu"
    
    SESSION_NAME="${gpu}_${tasks_1}"
    
    screen -dmS "$SESSION_NAME" bash -c "CUDA_VISIBLE_DEVICES=$gpu python mtl_main.py \
                                                    --dataset "mfrc" \
                                                    --label "Moral" \
                                                    --budget 0.25 \
                                                    --mtl_tasks "$tasks_1" \
                                                    --run_sweep \
                                                    --seed "0" ;
                                        CUDA_VISIBLE_DEVICES=$gpu python mtl_main.py \
                                                    --dataset "mfrc" \
                                                    --label "Moral" \
                                                    --budget 0.25 \
                                                    --mtl_tasks "$tasks_2" \
                                                    --run_sweep \
                                                    --seed "0" ;
                                        CUDA_VISIBLE_DEVICES=$gpu python mtl_main.py \
                                                   --dataset "mfrc" \
                                                    --label "Moral" \
                                                    --budget 0.25 \
                                                    --mtl_tasks "$tasks_3" \
                                                    --run_sweep \
                                                    --seed "0" ;
                                        CUDA_VISIBLE_DEVICES=$gpu python mtl_main.py \
                                                    --dataset "mfrc" \
                                                    --label "Moral" \
                                                    --budget 0.25 \
                                                    --mtl_tasks "$tasks_4" \
                                                    --run_sweep \
                                                    --seed "0" ;
                                        CUDA_VISIBLE_DEVICES=$gpu python mtl_main.py \
                                                   --dataset "mfrc" \
                                                    --label "Moral" \
                                                    --budget 0.25 \
                                                    --mtl_tasks "$tasks_5" \
                                                    --run_sweep \
                                                    --seed "0" ;
                                        "
done

# CUDA_VISIBLE_DEVICES=2 python mtl_main.py \
#                                                     --dataset "mfrc" \
#                                                     --label "Moral" \
#                                                     --budget 0.25 \
#                                                     --mtl_tasks "4,6,13,15,17,18" \
#                                                     --run_sweep \
#                                                     --dry_run \
#                                                     --seed "0" 