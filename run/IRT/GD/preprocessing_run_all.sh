#!/bin/bash
# Define subject arrays
math_subjects=("수학Ⅰ" "수학Ⅱ")
english_subjects=("독해")
korean_subjects=("독서" "문학")
satam_subjects=("한국사" "생활과 윤리" "정치와 법" "한국지리" "세계지리" "사회?문화" "동아시아사" "세계사" "윤리와 사상" "경제" "통합사회") 
gwatam_subjects=("물리학Ⅰ" "화학Ⅰ" "생명과학Ⅰ" "지구과학Ⅰ" "지구과학Ⅱ" "생명과학Ⅱ" "화학Ⅱ" "물리학Ⅱ" "통합과학")
math_combined="$(IFS=,; echo "${math_subjects[*]}")"
korean_combined="$(IFS=,; echo "${korean_subjects[*]}")"
english_combined="$(IFS=,; echo "${english_subjects[*]}")"
subjects=("$english_combined" "$korean_combined" "$math_combined" "${gwatam_subjects[@]}" "${satam_subjects[@]}" )
# subjects=("지구과학Ⅰ") 
mkdir -p detailed
mkdir -p processed

# .을 ?로 replace
python3 preprocessing_replace?.py

subjects_with_errors=()
subjects_without_errors=()
for subject in "${subjects[@]}"; do
    if [[ "$subject" == "was 윤리와 사상 before" ]]; then
        # 이 윤리와 사상일 때는 다른 파일을 돌았었는데, 그 이유는 
        # middle_freq_list.json 파일이 다음과 같이 한 소문제에서 여러개의 문제를 뽑아야 했기 때문이다.
        # '서양 윤리 사상': {
        #     '서양의 여러 가지 윤리 사상': 3}
        #
        # 근데, '서양의 여러 가지 윤리 사상' 안에는 소분류가 2개 밖에 없어서 3개를 뽑을 수 없었다.
        # 따라서 3으로 써져 있었지만, 2개만 뽑는 process를 다시 만드는 코드가 필요했다.
        #
        # 앞으로도 이런 경우가 생긴다면 이 코드를 다시 돌려야 한다. 하지만 그러지 않도록 제발 설득해라.
        
        if python3 preprocessing_ethicsNphilosophy.py --subjects "$subject"; then
            subjects_without_errors+=("$subject")
        else
            subjects_with_errors+=("$subject")
        fi
    else
        if python3 preprocessing.py --subjects "$subject"; then
            subjects_without_errors+=("$subject")
        else
            subjects_with_errors+=("$subject")
        fi
    fi
done

# Print the list of subjects with errors
if [ ${#subjects_with_errors[@]} -ne 0 ]; then
    echo "Subjects with assertion errors:"
    for subject in "${subjects_with_errors[@]}"; do
        echo "$subject"
    done
else
    echo "No subjects had assertion errors."
fi

# Print the list of subjects without errors
if [ ${#subjects_without_errors[@]} -ne 0 ]; then
    echo "Subjects without any problems:"
    for subject in "${subjects_without_errors[@]}"; do
        echo "$subject"
    done
else
    echo "No subjects processed without errors."
fi