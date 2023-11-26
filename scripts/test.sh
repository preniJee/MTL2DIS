# screen -dmS "ali" bash -c "echo $PATH ;
# sleep 10
# ;
# echo $PATH
# ;"


cd ..
source venv/bin/activate

GPUs=(3 4)

annotators_1=("Ann1,Ann2,Ann3" "Ann1,Ann2,Ann4" "Ann1,Ann2,Ann5" "Ann1,Ann2,Ann6" "Ann1,Ann3,Ann4")
annotators_2=("Ann1,Ann3,Ann5" "Ann1,Ann3,Ann6" "Ann1,Ann4,Ann5" "Ann1,Ann4,Ann6" "Ann1,Ann5,Ann6")
annotators_3=("Ann2,Ann3,Ann4" "Ann2,Ann3,Ann5" "Ann2,Ann3,Ann6" "Ann2,Ann4,Ann5" "Ann2,Ann4,Ann6")
annotators_4=("Ann2,Ann5,Ann6" "Ann3,Ann4,Ann5" "Ann3,Ann4,Ann6" "Ann3,Ann5,Ann6" "Ann4,Ann5,Ann6")
# Calculate the midpoint index for parallel execution
# midpoint=$((${#annotators[@]} / 2))

# Loop over the first half of tasks and run in parallel
for ann_index in  "${!annotators_1[@]}" ; do
    
    tasks_1=${annotators_1[$ann_index]}
    tasks_2=${annotators_2[$ann_index]}
    tasks_3=${annotators_3[$ann_index]}
    tasks_4=${annotators_4[$ann_index]}
    echo "heyy $tasks"
    # Get current gpu
    gpu_index=$(($ann_index % ${#GPUs[@]}))
    gpu=${GPUs[$gpu_index]}
    
    echo "Running model for tasks: $tasks_1, gpu: $gpu"
    
    SESSION_NAME="${gpu}_${tasks_1}"
    
    screen -dmS "$SESSION_NAME" bash -c "echo "$tasks_1";
                                        sleep 10;
                                        echo "$tasks_2";
                                        echo "$tasks_3";
                                        echo "$tasks_4";"
                                        
done
