import pandas as pd
all_subjects = [ '통합사회', '통합과학', '수학Ⅰ,수학Ⅱ', '독해,듣기', '독서,문학', '물리학Ⅰ', '화학Ⅰ', '생명과학Ⅰ', '지구과학Ⅰ', '지구과학Ⅱ', '생명과학Ⅱ', '화학Ⅱ', '물리학Ⅱ', '한국사', '생활과 윤리', '정치와 법', '한국지리', '세계지리', '사회?문화', '동아시아사', '세계사', '윤리와 사상', '경제' ]
sorted_df = pd.read_csv('/root/jw/EduCDM/examples/IRT/GD_inference/integration_test/compiled_data_with_penalty_0.3.csv', encoding='utf-8-sig')

correct_indexes = [1, 3, 6, 9, 10] # 1~10 까지의 문제 중 1, 3, 6, 9, 10번 문제를 맞춘 경우
all_subjects = [
    '수학Ⅰ,수학Ⅱ', '독해,듣기',  '지구과학Ⅰ', '생명과학Ⅰ', '독서,문학', '사회?문화', '생활과 윤리',
    '통합사회', '통합과학', '물리학Ⅰ', '화학Ⅰ', '지구과학Ⅱ', '생명과학Ⅱ', '화학Ⅱ', '물리학Ⅱ', 
    '한국사',  '정치와 법', '한국지리', '세계지리', 
    '동아시아사', '세계사', '윤리와 사상', '경제'
]

solving_history = [1 if i + 1 in correct_indexes else 0 for i in range(10)]
filtered_df = sorted_df[sorted_df['Solving History'].apply(lambda x: x == str(solving_history))]
result_df = pd.DataFrame({
    "Subject": all_subjects,
    "Rank": ["N/A"] * len(all_subjects),
    "Target Score": ["N/A"] * len(all_subjects)
})

for i, row in filtered_df.iterrows():
    subject = row["Subject"]
    rank = row["Rank"]
    target_score = row["Target Score"]
    result_df.loc[result_df["Subject"] == subject, ["Rank", "Target Score"]] = rank, target_score

# Save to CSV with solving history as filename
file_name = "solving_history_" + "_".join(map(str, solving_history)) + ".csv"
result_df.to_csv(file_name, index=False, encoding='utf-8-sig')

# print(result_df)
