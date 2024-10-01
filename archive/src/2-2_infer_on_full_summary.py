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
    
def greedy_inference(df, client):

    print("Infer for T")
    pbar = tqdm(total=df.shape[0])
    prompt = basic_prompt_t14

    ans_from_text_lst = []
    ans_from_abstract_lst = []
    ans_from_extract_lst = []
    
    for _, row in df.iterrows():

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

    df["zs_ans_from_text_t"] = ans_from_text_lst
    df["zs_ans_from_abstract_t"] = ans_from_abstract_lst
    df["zs_ans_from_extract_t"] = ans_from_extract_lst

    print("Infer for N")
    pbar = tqdm(total=df.shape[0])
    prompt = basic_prompt_n03

    ans_from_text_lst = []
    ans_from_abstract_lst = []
    ans_from_extract_lst = []
    
    for _, row in df.iterrows():

        if (type(row['n']) == int) or (type(row['n']) == float):
            prompt_template = MED42_PROMPT_TEMPLATE.format(system_instruction=system_instruction, prompt=prompt.format(report=row['text']))
            decoded_ans = client.text_generation(prompt=prompt_template, do_sample=False, max_new_tokens=12)
            ans_from_text_lst.append(decoded_ans)
            
            prompt_template = MED42_PROMPT_TEMPLATE.format(system_instruction=system_instruction, prompt=prompt.format(report=row['abstractive_summary']))
            decoded_ans = client.text_generation(prompt=prompt_template, do_sample=False, max_new_tokens=12)
            ans_from_abstract_lst.append(decoded_ans)

            prompt_template = MED42_PROMPT_TEMPLATE.format(system_instruction=system_instruction, prompt=prompt.format(report=row['extractive_summary']))
            decoded_ans = client.text_generation(prompt=prompt_template, do_sample=False, max_new_tokens=12)
            ans_from_extract_lst.append(decoded_ans)
        
        else:
            ans_from_text_lst.append("X")
            ans_from_abstract_lst.append("X")
            ans_from_extract_lst.append("X")

        pbar.update(1)
    pbar.close()

    df["zs_ans_from_text_n"] = ans_from_text_lst
    df["zs_ans_from_abstract_n"] = ans_from_abstract_lst
    df["zs_ans_from_extract_n"] = ans_from_extract_lst

    return df


if __name__ == "__main__":
    start_time = time.time()
    client = InferenceClient(model="http://127.0.0.1:8080")

    file_dir = "/secure/shared_data/rag_tnm_results/summary"
    file_name = 'withSummary_merged_df.csv'

    open_file_path = os.path.join(file_dir,file_name)
    save_file_path = os.path.join(file_dir, "withInference_"+ file_name)
    df = pd.read_csv(open_file_path)
    df = greedy_inference(df, client)
    df.to_csv(save_file_path, index=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
