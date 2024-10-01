import pandas as pd
from huggingface_hub import InferenceClient
from tqdm import tqdm
import glob
import os
from collections import defaultdict
import argparse

def greedy_inference(df, client):
    """
    perform greedy inference using InferenceClient.text_generation API
    greedy inference is enabled by setting do_sample=False
    """

    pbar = tqdm(total=df.shape[0])

    ans_list = []

    for _, row in df.iterrows():

        # format the prompt
        formatted_prompt = row["formatted_fs_prompts"]
        # call the text generation API
        decoded_ans = client.text_generation(prompt=formatted_prompt, do_sample=False, 
                                             max_new_tokens=12)
        ans_list.append(decoded_ans)

        pbar.update(1)
    pbar.close()

    df["ans_str"] = ans_list
    return df

def sampling_inference(df, client, temperature, top_p,  num_return_sequences):
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
        formatted_prompt = row["formatted_fs_prompts"]
        
        pass
        
        pbar.update(1)
    pbar.close()

    #for key in sampling_ans_dict.keys():
    #    col_name = "ans_str_{}".format(key)
    #    df[col_name] = sampling_ans_dict[key]
        
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-d", "--data", help="the path of test csv data",
    #                     type=str, required=True)
    # parser.add_argument("-o", "--output", help="the path of output csv data",
    #                     type=str, required=True)
    parser.add_argument("-dec", "--decoding", help="the decoding method (greedy or sampling)",
                        type=str, required=True)
    parser.add_argument("-nrs", "--num_return_sequences", help="num_return_sequences (required if using sampling)",
                        type=int)
    parser.add_argument("-t", "--temperature", help="the temperature for sampling (required if using sampling)",
                        type=float)
    parser.add_argument("-tp", "--top_p", help="parameter for nucleus sampling (required if using sampling)",
                        type=float)
    args = parser.parse_args()
    
    if args.decoding == "sampling":
        raise ValueError(f"""Sampling inference is not implemented now!.""")
        
    # initialized client
    client = InferenceClient(model="http://127.0.0.1:8080")
    for file_path in glob.glob(os.path.join("/home/yl3427/cylab/yewon_data/", "*_fold_test_split.csv")):
        df = pd.read_csv(file_path)

    # load test data stored in csv file
    # df = pd.read_csv(args.data)

        if args.decoding == "greedy":
            # perform greedy inference
            df = greedy_inference(df, client)
        elif args.decoding == "sampling":
            # perform sampling inference 
            df = sampling_inference(df, client, \
                                    args.temperature, args.top_p, args.num_return_sequences)

        # save_file = args.output
        base_name = os.path.basename(file_path)
        save_file = os.path.join("/home/yl3427/cylab/yewon_data/", "result_" + base_name)
        print("Check {} for results.".format(save_file))
        df.to_csv(save_file, index=False)