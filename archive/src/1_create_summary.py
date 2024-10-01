from huggingface_hub import InferenceClient
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import os
import re

# agnostic to label, cancer types

# prompts
MED42_PROMPT_TEMPLATE = """
<|system|>:{system_instruction}
<|prompter|>:{prompt}
<|assistant|>:
"""

system_instruction_for_abstract = """You are an expert at interpreting and summarizing pathology reports for cancer staging purposes.
Summarize the following pathological report focusing on details pertinent to the pathological staging of the patient's cancer, including tumor size, lymph node involvement, and metastasis status.
Your summary should provide clear information to aid in the pathological staging of the patient’s cancer.
Limit your summary to 100 words and format it as: 'Summary: [Your summary]'.
Provide only the required summary without any additional text or commentary."""

system_instruction_for_extract= """You are an expert at interpreting pathology reports for cancer staging purposes.
Your task is to extract and present key information directly from the following pathological report. Focus specifically on details pertinent to the pathological staging of the patient's cancer, such as tumor size, lymph node involvement, and metastasis status. 
Your summary should not exceed 100 words and should format as: 'Summary: [Your extractive summary]'.
Refrain from paraphrasing or adding any additional text or commentary; your summary should consist solely of relevant excerpts from the report."""

prompt = """Report:{report}
Summary: """
  

def greedy_inference_summary(df, client):

    pbar = tqdm(total=df.shape[0])

    abstract_lst = []
    extract_lst = []

    for _, row in df.iterrows():
        # ['patient_filename', 't', 'text', 'prompt_for_abstract','prompt_for_extract']
        formatted_prompt = row["prompt_for_abstract"]
        decoded_ans = client.text_generation(prompt=formatted_prompt, do_sample=False,
                                             max_new_tokens=256)
        abstract_lst.append(decoded_ans)

        formatted_prompt = row["prompt_for_extract"]
        decoded_ans = client.text_generation(prompt=formatted_prompt, do_sample=False,
                                             max_new_tokens=256)
        extract_lst.append(decoded_ans)

        pbar.update(1)
    pbar.close()

    df["abstractive_summary"] = abstract_lst
    df["extractive_summary"] = extract_lst

    return df

if __name__ == "__main__":
    start_time = time.time()

    client = InferenceClient(model="http://127.0.0.1:8080")

    # input
    file_dir = "/secure/shared_data/rag_tnm_results/summary"
    # file_names = ["BRCA_N03_testing.csv", "BRCA_T14_testing.csv", "LUAD_N03_testing.csv", "LUAD_T14_testing.csv",
    # "BRCA_N03_training.csv", "BRCA_T14_training.csv", "LUAD_N03_training.csv", "LUAD_T14_training.csv"]
    file_names = ['merged_df.csv']

    for file_name in file_names:
        open_file_path = os.path.join(file_dir, file_name)
        save_file_path = os.path.join(file_dir, "withSummary_"+file_name)

        df = pd.read_csv(open_file_path)

        abstractive_prompts = []
        extractive_prompts = []

        for _, each_report in df.iterrows():
            formatted_prompt = prompt.format(report=each_report["text"])
            model_formatted_prompt = MED42_PROMPT_TEMPLATE.format(system_instruction=system_instruction_for_abstract,
                                                            prompt=formatted_prompt)
            abstractive_prompts.append(model_formatted_prompt)

            model_formatted_prompt = MED42_PROMPT_TEMPLATE.format(system_instruction=system_instruction_for_extract,
                                                            prompt=formatted_prompt)
            extractive_prompts.append(model_formatted_prompt)
            
        df["prompt_for_abstract"] = abstractive_prompts
        df["prompt_for_extract"] = extractive_prompts
        
        df = greedy_inference_summary(df, client)
        df.to_csv(save_file_path, index=False)

        

    