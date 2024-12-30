#!/bin/bash

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
subjects=("$math_combined" "$english_combined" "$korean_combined" "${gwatam_subjects[@]}" "${satam_subjects[@]}" )
# subjects=("$english_combined")
# subjects=("독해")


# Prepare results directory
mkdir -p results_NA2

# Initialize best score storage
declare -A best_scores
for subject in "${subjects[@]}"; do
    best_scores["$subject"]="Initial"
    mkdir -p "results_NA2/$subject"
done

for subject in "${subjects[@]}"; do
    echo "Processing subject: $subject"
    output=$(python IRT_mine_NA2_useOriginal.py --subjects "$subject")
    echo $output
    current_score=$(echo "$output" | grep 'Combined Mean Score' | awk -F ':' '{print $2}' | tr -d ',' | awk '{print $1}' | xargs)
    mean_auc=$(echo "$output" | grep 'mean auc score' | awk -F 'is ' '{print $2}')
    mean_acc=$(echo "$output" | grep 'mean acc score' | awk -F 'is ' '{print $2}')
    mean_fpr=$(echo "$output" | grep 'mean fpr score' | awk -F 'is ' '{print $2}')
    mean_recall=$(echo "$output" | grep 'mean recall score' | awk -F 'is ' '{print $2}')
    # Collect and save best parameters
    best_params=""
    best_params+="Combined Mean Score: $current_score\n"
    best_params+="Mean AUC Score: $mean_auc\n"
    best_params+="Mean ACC Score: $mean_acc\n"
    best_params+="Mean FPR Score: $mean_fpr\n"
    best_params+="Mean Recall Score: $mean_recall\n"
    echo "End for $subject: Best Score so far: ${best_params}"
    best_scores[$subject]=$best_params
    echo -e "$best_params" > "results_NA2/$subject/{$subject}_NA2_IRT_Original_results.txt"

done

for subject in "${subjects[@]}"; do
    echo "Final best parameters for $subject: ${best_scores[$subject]}"
done
