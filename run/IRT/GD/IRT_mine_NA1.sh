#!/bin/bash

# Define subject arrays
math_subjects=("수학Ⅰ" "수학Ⅱ")
english_subjects=("독해")
korean_subjects=("독서" "문학")
# Define other subjects
satam_subjects=("한국사" "생활과 윤리" "정치와 법" "한국지리" "세계지리" "사회?문화" "동아시아사" "세계사" "윤리와 사상" "경제")
gwatam_subjects=("물리학Ⅰ" "화학Ⅰ" "생명과학Ⅰ" "지구과학Ⅰ" "지구과학Ⅱ" "생명과학Ⅱ" "화학Ⅱ" "물리학Ⅱ")

# Combine subjects
math_combined="$(IFS=,; echo "${math_subjects[*]}")"
korean_combined="$(IFS=,; echo "${korean_subjects[*]}")"
english_combined="$(IFS=,; echo "${english_subjects[*]}")"

# subjects=("$math_combined" "$english_subjects" "$korean_combined" "${gwatam_subjects[@]}" "${satam_subjects[@]}")
subjects=("동아시아사" "$korean_combined" "물리학Ⅱ")


# Prepare results directory
mkdir -p results_NA1
timestamp=$(date +%Y%m%d_%H%M%S)

# Define simulation parameters
total_cycles=1000000
iterations_per_cycle=50
trial=10

# Initialize best score storage
declare -A best_scores
for subject in "${subjects[@]}"; do
    best_scores["$subject"]=0
    mkdir -p "results_NA1/$subject"
done

# Main simulation loop
for (( cycle = 0; cycle < total_cycles; cycle++ )); do
    for subject in "${subjects[@]}"; do
        echo "Processing subject: $subject"
        for (( i = 0; i < iterations_per_cycle; i++ )); do
            # Generate random parameters
            alpha=$(python -c "import random; print(round(random.uniform(0.1, 0.9), 2))")
            gamma=$(python -c "import random; print(random.choice([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5.0]))")
            batch_size=$(python -c "import random; print(random.choice([32, 64, 128, 256, 512, 1024]))")

            # Simulation parameters
            epochs=2000
            patience=50
            seed=42
            param_type=3
            train_ratio=0.7

            # Execute Python script
            output=$(python IRT_mine_NA1.py --trial $trial --timestamp "$timestamp" --subjects "$subject" --alpha $alpha --gamma $gamma --batch_size $batch_size --epochs $epochs --patience $patience --seed $seed --param_type $param_type --train_ratio $train_ratio)

            # Parse output for scores
            current_score=$(echo "$output" | grep 'Combined Mean Score' | awk -F ':' '{print $2}' | tr -d ',' | awk '{print $1}' | xargs)

            # Update best score if current score is higher
            if [[ $(echo "$current_score > ${best_scores[$subject]}" | bc -l) -eq 1 ]]; then
                best_scores["$subject"]=$current_score
                mean_auc=$(echo "$output" | grep 'mean auc score' | awk -F 'is ' '{print $2}')
                mean_acc=$(echo "$output" | grep 'mean acc score' | awk -F 'is ' '{print $2}')
                mean_fpr=$(echo "$output" | grep 'mean fpr score' | awk -F 'is ' '{print $2}')
                mean_recall=$(echo "$output" | grep 'mean recall score' | awk -F 'is ' '{print $2}')

                # Collect and save best parameters
                best_params="Cycle: $cycle, Iteration: $i, Alpha: $alpha, Gamma: $gamma, Batch Size: $batch_size, Epochs: $epochs, Patience: $patience, Seed: $seed, Param Type: $param_type, Train Ratio: $train_ratio\n"
                best_params+="Combined Mean Score: $current_score\n"
                best_params+="Mean AUC Score: $mean_auc\n"
                best_params+="Mean ACC Score: $mean_acc\n"
                best_params+="Mean FPR Score: $mean_fpr\n"
                best_params+="Mean Recall Score: $mean_recall\n"

                echo -e "$best_params" > "results_NA1/$subject/best_params_${timestamp}.txt"

            fi
        done
        echo "End of cycle $cycle for $subject: Best Score so far: ${best_scores[$subject]}"
    done
done

# Final output after all cycles
for subject in "${subjects[@]}"; do
    echo "Final best parameters for $subject: ${best_scores[$subject]}"
done
