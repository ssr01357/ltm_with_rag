import chromadb
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
import tiktoken
from collections import defaultdict
from tqdm import tqdm
import time
import re
import os
import pickle

# prompts
MED42_PROMPT_TEMPLATE = """<|system|>:{system_instruction}
<|prompter|>:{prompt}
<|assistant|>:"""

SHOT_TEMPLATE = """
Report:{report}
Answer:{label}"""

system_instruction_short_ans_t14 = "You are an expert at interpreting pathology reports for cancer staging. You provide only one word answers. You must answer from the following four options:  T1, T2, T3, T4.  Here is an example of a well formatted answer: 'Ti' where i is from 1 to 4. You do not output any additional text."
system_instruction_short_ans_n03 = "You are an expert at interpreting pathology reports for cancer staging. You provide only one word answers. You must select from the following four options:  N0, N1, N2, N3. Here is an example of a well formatted answer: 'Ni' where i is from 0 to 3. You do not output any additional text."


dfs_prompt_t14 = """Using the following reports as a guide, your task is to review the provided pathology report for a cancer patient and determine what the T stage is from this report. Ignore any substaging information.  You must select from the following four options:  T1, T2, T3, T4.
Only provide the T stage with no additional text.  
{demonstrations} 

Report:{report}
Answer: 
"""
dfs_prompt_n03 = """Using the following reports as a guide, your task is to review the provided pathology report for a cancer patient and determine what the N stage is from this report. Ignore any substaging information.  You must select from the following four options:  N0, N1, N2, N3.
Only provide the N stage with no additional text.
{demonstrations} 

Report:{report}
Answer: 
"""

# functions
def count_tokens(text, tokenizer):
    input_ids = tokenizer(text, return_tensors='pt').input_ids.cuda()
    start_index = input_ids.shape[-1]
    return start_index


client = chromadb.PersistentClient(path="/home/yl3427/cylab/chroma_db")
model_name_or_path = "m42-health/med42-70b"
cache_dir = "/secure/chiahsuan/hf_cache"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
top_n = 1
k = 1 # number of shot for each category

#  input
test_dir = "/home/yl3427/cylab/yewon_data"
test_file_name = "usingExtract_fs_merged_df.csv"

open_file_path = os.path.join(test_dir, test_file_name)



collection = client.get_collection("full_summary_emb_2") # full_report_emb, full_summary_emb, full_summary_emb_2

with open('/home/yl3427/cylab/rag_tnm/full_5folds_dict.pkl', 'rb') as file:
    loaded_dict = pickle.load(file)

df = pd.read_csv(open_file_path)

for mode, folds in loaded_dict.items():
    for idx, fold in enumerate(folds):
        print(f"{mode} {idx+1}st fold")
        label = mode.split("_")[1]
        list_of_test_ids = fold['test']
        test_df = df[df['patient_filename'].isin(list_of_test_ids)]
        dynamic_few_shots = defaultdict(list)
        pbar = tqdm(total=test_df.shape[0])
        for _, report in test_df.iterrows():
            test_return_obj = collection.get(ids=report["patient_filename"], include=["embeddings", "metadatas"])
            test_report_embedding = test_return_obj["embeddings"]

            for possible_label in [0,1,2,3]:
                # if (label == 'n') and (idx == 0) and (mode.split("_")[0] == "luad") and (possible_label == 3):
                #     shot_key = "dfs_{}{}_0".format(label, possible_label) 
                #     dynamic_few_shots[shot_key].append("Tumour extends beyond. the muscularis propria into the adventitia but does not extend to. the circumferential margin. Tumour is present in 12 out of the 18. nodes identified. 2-4. Sections of the splenic, subcarinal and right cross lymph. nodes show no evidence of invasive malignancy. Pathological staging:TNM 7 T3 N3 Mx. Signature of Pathologist (s).")
                #     break;
                retrieved_items = collection.query(query_embeddings=test_report_embedding,
                                                           include=["embeddings", "metadatas", "documents"],
                                                           n_results=1,
                                                           where={"$and":
                                                                  [{"patient_filename": {'$in': fold['train']}},
                                                                   {f'is_goodsum_{label}':{"$eq":True}},
                                                                   {f'{label}_label':{"$eq":possible_label}}
                                                                   ]})
                # print(len(retrieved_items["documents"][0]))
                # assert len(retrieved_items["documents"][0])>1, mode + len(retrieved_items["documents"][0])
                # break
                doc = retrieved_items["documents"][0][0] # since top_n == 1, there is only one document
                # retrieved_items = collection.query(query_embeddings=test_report_embedding,
                #                                            include=["embeddings", "metadatas"],
                #                                            n_results=top_n,
                #                                            where={"$and":
                #                                                   [{"patient_filename": {'$in': fold['train']}},
                #                                                    {f'is_goodsum_{label}':{"$eq":True}},
                #                                                    {f'{label}_label':{"$eq":possible_label}}
                #                                                    ]})
                # doc = retrieved_items['metadatas'][0][0]['abstract']  
                shot_key = "dfs_{}{}_0".format(label, possible_label) 
                dynamic_few_shots[shot_key].append(doc)
            pbar.update(1)
        pbar.close()

        for key in dynamic_few_shots.keys():
            assert test_df.shape[0] == len(dynamic_few_shots[key])
            test_df[key] = dynamic_few_shots[key]

        col_key = "dfs_{}{}_{}"
        formatted_fs_prompts = []
        num_of_tokens = []

        for _, each_report in test_df.iterrows():
            demos = []
            for i in [0, 1, 2, 3]:
                if label == "n":
                    shot_label = "{}{}".format(label.upper(), i)
                elif label == "t":
                    shot_label = "{}{}".format(label.upper(), i+1) # i+1 is necessary for T category because 0 maps to T1, ..., 3 maps to T4
                for j in range(k): # top K
                    shot_report = each_report[col_key.format(label, i, j)]
                    demos.append(SHOT_TEMPLATE.format(report=shot_report, label=shot_label))
            demo_string = "\n".join(demos)
            if label == "t":
                formatted_prompt = dfs_prompt_t14.format(demonstrations=demo_string, report=each_report["text"])
                model_formatted_prompt = MED42_PROMPT_TEMPLATE.format(system_instruction=system_instruction_short_ans_t14,
                                                                prompt=formatted_prompt)
            elif label == "n":
                formatted_prompt = dfs_prompt_n03.format(demonstrations=demo_string, report=each_report["text"])
                model_formatted_prompt = MED42_PROMPT_TEMPLATE.format(system_instruction=system_instruction_short_ans_n03,
                                                                prompt=formatted_prompt)

            formatted_fs_prompts.append(model_formatted_prompt)
            num_of_tokens.append(count_tokens(model_formatted_prompt, tokenizer))
            
        test_df["formatted_fs_prompts"] = formatted_fs_prompts
        test_df["num_of_tokens"] = num_of_tokens

        save_file_path = os.path.join("/home/yl3427/cylab/yewon_data", f"{mode}_{idx+1}st_fold_test_split.csv")
        test_df.to_csv(save_file_path, index=False) 