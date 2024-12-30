import os
import json
import pandas as pd

directory = "../../../data/full_data"  # /root/jw/TAPTAP_IRT/data/full_data


#json
for filename in os.listdir(directory):
    if "middle_freq_list" in filename:
        filepath = os.path.join(directory, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        data_str = json.dumps(data, ensure_ascii=False)
        data_str = data_str.replace("・", "?").replace("∙", "?")
        modified_data = json.loads(data_str)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(modified_data, f, ensure_ascii=False, indent=4)
#CSV
for filename in os.listdir(directory):
    if "cumul" in filename:
        filepath = os.path.join(directory, filename)
        try:
            df = pd.read_csv(filepath, encoding="utf-8")
        except:
            df = pd.read_csv(filepath, encoding='cp949') 
        df.replace(to_replace=["・", "∙"], value="?", regex=True, inplace=True)
        df.to_csv(filepath, index=False, encoding="utf-8")
        
print("Replacement complete.")
