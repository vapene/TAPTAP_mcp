
# **기출탭탭 IRT - Instruction Guide**  
**Inference와 통합 테스트에 관한 문서**

---

## **Getting Started**

### **0. Prerequisites**  
Ensure you are in the correct directory before beginning the process:

```bash
conda activate lanchain
cd /root/jw/TAPTAP_IRT/run/IRT/GD_inference
```

---

### **0-1. Data Preparation**  
**Data files** are located at:  
`/root/jw/TAPTAP_IRT/run/IRT/GD`  

| File                                | Description                                            |
|-------------------------------------|--------------------------------------------------------|
| `results_NA1/{subject}/irt.params`  | Parameters for IRT                                     |
| `configs_NA1.yml`                   | Best hyper-parameters                                 |
| `processed/10_{args.subjects}_30_ratio_afterNA1.csv` | Student's solving history                    |

**Criteria for non-IRT subjects**:  
Located at: `/root/jw/TAPTAP_IRT/run/IRT/GD/non_IRT_subjects_criteria/criteria_output`  

- `{한국사}_results.json`: Lower and upper threshold for each subject.  

👉 **Note:** Manually create `middle_high_rank.json` in the following path using the above dataset:  
`/root/jw/TAPTAP_IRT/run/IRT/GD_inference/middle_high_rank.json`  

---

### **0-2. Single Subject IRT Execution**  
단일 subject에 대한 IRT 결과가 궁금하다면:  

Run the **`incremental_inference.py`** script to get:  
```
Prints (rank, '상중하', target_score, '능력치')
```

---

## **1. Get results for all solving history**  
Execute `Integration_test_with_consist_penalty.py`
**Inputs from GD:**  
1. `configs_NA1.yml`  
2. `irt.params`  
3. `10_{subjects}_problem_point_mapping_dict`  

**Additional Input:**  
- `middle_high_rank_with_penalty.json`  

**Output:**  
- **integration_test/compiled_data.csv**  
모든 과목의 

| Columns                       | Example                                                |
|-------------------------------|--------------------------------------------------------|
| Subject                       | `한국사`                                               |
| Rank                          | `middle`                                               |
| Solving History               | `"[1, 1, 1, 1, 0, 0, 0, 1, 0, 0]"`                     |
| Target Score                  | `13.0`                                                 |

---

## **2. Get Threshold for IRT**  

Execute the code **`IRT_threshold.py`**.

**Inputs:**  
1. `irt_subjects list`  
2. `GD/detailed/10_{subject}_p_theta_response.json`  
3. `integration_test/compiled_data.csv`  

**Output Example:**  
```json
"독서,문학": [
    -0.3161628246307373,
    0.0112729333341121
]
```

👉 **Note:** Manually update the results of `middle_high_rank.json` for IRT subjects.

## **3. Rerun `integration_test_with_consist_penalty.py`**

Re-run with updated `middle_high_rank.json` will give the final `compiled_data.csv`.

---

## **통합 테스트**  
📁 Look into the folder **`test_scripts`** for post-hoc analysis.

각 문제를 맞출 확률 p_{question}을 가지고 simulation 해서 만든 값에 상,중,하를 30%씩 배분했다. 
따라서, 10문제 중 약 70%를 맞춰야만 '중'이 뜰 것이다.  

---

### **1. Get Answers for Each Subject**  

Execute the script: **`get_answers.py`**  

**Output:**  
- Generates: `GD_inference/integration_test/filtered_csv_outputs` for each subject.

---

### **2. Get Results for All Subjects by Solving History**  

Run the script: **`get_rank_by_response.py`**

**Output:**  
Generates:  
`solving_history_" + "_".join(map(str, solving_history)) + ".csv`  

**Example Result:**  

| Subject            | Rank  | Target Score              |
|---------------------|-------|---------------------------|
| `수학Ⅰ,수학Ⅱ`      | `low` | `-1.1868665218353271`     |
| `독해,듣기`         | `low` | `-0.7025999426841736`     |

---

**End of Document** 🚀  
