from archive.prompts import system_instruction, basic_prompt, few_shots_prompt, basic_prompt_t14, basic_prompt_n03

import pandas as pd
from huggingface_hub import InferenceClient
from tqdm import tqdm

from collections import defaultdict
import argparse
import os

MED42_PROMPT_TEMPLATE = """
<|system|>:{system_instruction}
<|prompter|>:{prompt}
<|assistant|>:
"""

LLAMA_PROMPT_TEMPLATE = """
<s>[INST] <<SYS>>
{system_instruction}
<</SYS>>

{prompt} [/INST]
answer:
"""

CLINICALCAMEL_PROMPT_TEMPLATE = """
system: {system_instruction}
question: {prompt}
answer:
"""

def _format_prompt_function(text, prompting, model_name_or_path, data):
    # filled the prompt based on the prompting strategy
    if prompting == "zs":
        if "m01" in data:
            prompt = basic_prompt.format(report=text)
        elif "n03" in data:
            prompt = basic_prompt_n03.format(report=text)
        elif "t14" in data:
            prompt = basic_prompt_t14.format(report=text)
    elif prompting == "fs":
        # currently few-shots only support m01
        prompt = few_shots_prompt.format(report=text)
    else:
        raise ValueError(f"""{prompting} is not supported.""")

    if "med42" in model_name_or_path:
        return MED42_PROMPT_TEMPLATE.format(system_instruction=system_instruction, prompt=prompt)
    elif "Llama-2" in model_name_or_path:
        return LLAMA_PROMPT_TEMPLATE.format(system_instruction=system_instruction, prompt=prompt)
    elif "ClinicalCamel" in model_name_or_path:
        return CLINICALCAMEL_PROMPT_TEMPLATE.format(system_instruction=system_instruction, prompt=prompt)
    else:
        raise ValueError(f"""{model_name_or_path} is not supported.""")

def create_save_file_path(**kwargs):
    dirname = os.path.dirname(kwargs["data"])
    model_name_or_path = kwargs["model_name_or_path"]
    
    if "med42" in model_name_or_path:
        model_abbr = "med42"
    elif "Llama-2" in model_name_or_path:
        model_abbr = "Llama-2"
    elif "ClinicalCamel" in model_name_or_path:
        model_abbr = "ClinicalCamel"
    
    if kwargs["decoding"] == "greedy":
        save_file_name = "{}-{}.csv".format(model_abbr, kwargs["prompting"])
        return os.path.join(dirname, "greedy", save_file_name)
    elif kwargs["decoding"] == "sampling":
        save_file_name = "{}-{}-t{}-tp{}-nrs{}.csv".format(model_abbr, kwargs["prompting"], kwargs["temperature"],
                                                           kwargs["top_p"], kwargs["num_return_sequences"])
        return os.path.join(dirname, "sampling", save_file_name)

def greedy_inference(df, client, prompting, model_name_or_path, data):
    """
    perform greedy inference using InferenceClient.text_generation API
    greedy inference is enabled by setting do_sample=False
    """

    pbar = tqdm(total=df.shape[0])

    ans_list = []

    for _, row in df.iterrows():

        # format the prompt
        prompt_template = _format_prompt_function(row['text'], prompting, model_name_or_path, data)
        #print(prompt_template)
        # call the text generation API
        decoded_ans = client.text_generation(prompt=prompt_template, do_sample=False, 
                                             max_new_tokens=12)
        ans_list.append(decoded_ans)

        pbar.update(1)
    pbar.close()

    df["ans_str"] = ans_list
    return df

def sampling_inference(df, client, prompting, model_name_or_path, temperature, top_p,  num_return_sequences, data):
    """
    this function need to be modified later. DO NOT USE IT. 

    perform sampling inference using InferenceClient.text_generation API
    sampling inference is enabled by setting do_sample=True
    the sampling inference requires temperature, nucleus parameter (top_p), and number of return sequences
    """

    pbar = tqdm(total=df.shape[0])

    sampling_ans_dict = defaultdict(list)

    for _, row in df.iterrows():

        # format the prompt
        prompt_template = _format_prompt_function(row['text'], prompting, model_name_or_path, data)
        
        pass
        
        pbar.update(1)
    pbar.close()

    #for key in sampling_ans_dict.keys():
    #    col_name = "ans_str_{}".format(key)
    #    df[col_name] = sampling_ans_dict[key]
        
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-g", "--gpu_index", help="two gpu index in the server, e.g., 4,5",
    #                     type=str, required=True)
    parser.add_argument("-m", "--model_name_or_path", help="model name listed in cache directory",
                        type=str, required=True)
    # parser.add_argument("-c", "--cacheDir", help="the path of cache directory",
    #                     default="/secure/chiahsuan/hf_cache", type=str)
    parser.add_argument("-d", "--data", help="the path of test csv data",
                        type=str, required=True)
    parser.add_argument("-p", "--prompting", help="the prompting strategy, e.g., zero-shot (zs) or few-shots (fs)",
                        type=str, required=True)
    parser.add_argument("-dec", "--decoding", help="the decoding method (greedy or sampling)",
                        type=str, required=True)
    parser.add_argument("-nrs", "--num_return_sequences", help="num_return_sequences (required if using sampling)",
                        type=int)
    parser.add_argument("-t", "--temperature", help="the temperature for sampling (required if using sampling)",
                        type=float)
    parser.add_argument("-tp", "--top_p", help="parameter for nucleus sampling (required if using sampling)",
                        type=float)
    args = parser.parse_args()
    
    if args.prompting == "fs":
        if "m01_data" not in args.data:
            raise ValueError(f"""few-shots prompting only supporte m01 data now!.""")
        
    # initialized client
    client = InferenceClient(model="http://127.0.0.1:8080")

    # load test data stored in csv file
    df = pd.read_csv(args.data)

    if args.decoding == "greedy":
        # perform inference
        df = greedy_inference(df, client, args.prompting, args.model_name_or_path, args.data)
    
        # make path for save file
        save_file = create_save_file_path(data=args.data, model_name_or_path=args.model_name_or_path,\
                                      prompting=args.prompting, decoding=args.decoding)
    elif args.decoding == "sampling":
        # perform inference 
        df = sampling_inference(df, client, args.prompting, args.model_name_or_path,\
                                args.temperature, args.top_p, args.num_return_sequences, args.data)
    
        # make path for save file
        save_file = create_save_file_path(data=args.data, model_name_or_path=args.model_name_or_path,\
                                      prompting=args.prompting, decoding=args.decoding, temperature=args.temperature,\
                                      top_p=args.top_p, num_return_sequences=args.num_return_sequences)
        
    print(save_file)
    df.to_csv(save_file)