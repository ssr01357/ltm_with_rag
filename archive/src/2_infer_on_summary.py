from archive.prompts import system_instruction, basic_prompt_t14, basic_prompt_n03
import pandas as pd
from huggingface_hub import InferenceClient
from tqdm import tqdm
import re
import os
import time

MED42_PROMPT_TEMPLATE = """
<|system|>:{system_instruction}
<|prompter|>:{prompt}
<|assistant|>:
"""
   
    
def greedy_inference(df, client, label):
    pbar = tqdm(total=df.shape[0])

    ans_from_text_lst = []
    ans_from_abstract_lst = []
    ans_from_extract_lst = []
    
    for _, row in df.iterrows():
        # patient_filename,m,text,prompt_for_abstract,prompt_for_extract,abstractive_summary,extractive_summary

        if label == "t":
            prompt = basic_prompt_t14
        elif label == "n":
            prompt = basic_prompt_n03

        prompt_template = MED42_PROMPT_TEMPLATE.format(system_instruction=system_instruction, prompt=prompt.format(report=row['text']))
        decoded_ans = client.text_generation(prompt=prompt_template, do_sample=False, max_new_tokens=12)
        ans_from_text_lst.append(decoded_ans)
        
        prompt_template = MED42_PROMPT_TEMPLATE.format(system_instruction=system_instruction, prompt=prompt.format(report=row['abstractive_summary']))
        decoded_ans = client.text_generation(prompt=prompt_template, do_sample=False, max_new_tokens=12)
        ans_from_abstract_lst.append(decoded_ans)

        prompt_template = MED42_PROMPT_TEMPLATE.format(system_instruction=system_instruction, prompt=prompt.format(report=row['extractive_summary']))
        decoded_ans = client.text_generation(prompt=prompt_template, do_sample=False, max_new_tokens=12)
        ans_from_extract_lst.append(decoded_ans)

        pbar.update(1)
    pbar.close()

    df["zs_ans_from_text"] = ans_from_text_lst
    df["zs_ans_from_abstract"] = ans_from_abstract_lst
    df["zs_ans_from_extract"] = ans_from_extract_lst
    

    return df



if __name__ == "__main__":
    start_time = time.time()
    client = InferenceClient(model="http://127.0.0.1:8080")

    file_dir = "/home/yl3427/rag_tnm2/summary/data/summary"
    file_names = ["BRCA_N03_testing.csv", "BRCA_T14_testing.csv", "LUAD_N03_testing.csv", "LUAD_T14_testing.csv",
    "BRCA_N03_training.csv", "BRCA_T14_training.csv", "LUAD_N03_training.csv", "LUAD_T14_training.csv"]

    for file_name in file_names:
        file_path = os.path.join(file_dir, "withSummary_"+file_name)

        if "n03" in re.search(r'[M|N|T]\d\d', file_name).group().lower():
            label = "n"
        elif "t14" in re.search(r'[M|N|T]\d\d', file_name).group().lower():
            label = "t"

        df = pd.read_csv(file_path)
        df = greedy_inference(df, client, label)
        df.to_csv(file_path, index=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

