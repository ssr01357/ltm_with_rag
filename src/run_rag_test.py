from agent import *
from prompt import *
from openai import OpenAI
import numpy as np
import pandas as pd
import os
import json

if __name__ == "__main__":
   with open("/home/yl3427/cylab/rag_tnm/src/context.json", "r") as f:
      context = json.load(f)

   rag_raw_t14 = context["rag_raw_t14"]
   rag_raw_n03 = context["rag_raw_n03"]
   ltm_zs_t14 = context["ltm_zs_t14"]
   ltm_zs_n03 = context["ltm_zs_n03"]
   ltm_rag1_t14 = context["ltm_rag1_t14"]
   ltm_rag1_n03 = context["ltm_rag1_n03"]
   ltm_rag2_t14 = context["ltm_rag2_t14"]
   ltm_rag2_n03 = context["ltm_rag2_n03"]

   agent = Agent()

   brca_df = pd.read_csv("/secure/shared_data/rag_tnm_results/summary/5_folds_summary/brca_df.csv")
   brca_df = brca_df[brca_df["n"]!=-1]
   brca_df = brca_df[['patient_filename', 't', 'text', 'n']]

   # Zero-shot Chain of Thought
   result_df = agent.get_tnm_stage(brca_df, "zscot", zscot_t14, "t")
   result_df = agent.get_tnm_stage(result_df, "zscot", zscot_n03, "n")
   result_df.to_csv("/home/yl3427/cylab/rag_tnm/rag_result/0929_zscot.csv", index=False)

   # RAG with raw chunks
   result_df = agent.get_tnm_stage(result_df, "rag_raw", rag_t14, "t", rag_raw_t14)
   result_df = agent.get_tnm_stage(result_df, "rag_raw", rag_n03, "n", rag_raw_n03)
   result_df.to_csv("/home/yl3427/cylab/rag_tnm/rag_result/0929_rag_raw.csv", index=False)

   # ltm with zero-shot
   result_df = agent.get_tnm_stage(result_df, "ltm_zs", ltm_t14, "t", ltm_zs_t14)
   result_df = agent.get_tnm_stage(result_df, "ltm_zs", ltm_n03, "n", ltm_zs_n03)
   result_df.to_csv("/home/yl3427/cylab/rag_tnm/rag_result/0929_ltm_zs.csv", index=False)

   # ltm with rag1
   result_df = agent.get_tnm_stage(result_df, "ltm_rag1", ltm_t14, "t", ltm_rag1_t14)
   result_df = agent.get_tnm_stage(result_df, "ltm_rag1", ltm_n03, "n", ltm_rag1_n03)
   result_df.to_csv("/home/yl3427/cylab/rag_tnm/rag_result/0929_ltm_rag1.csv", index=False)

   # ltm with rag2
   result_df = agent.get_tnm_stage(result_df, "ltm_rag2", ltm_t14, "t", ltm_rag2_t14)
   result_df = agent.get_tnm_stage(result_df, "ltm_rag2", ltm_n03, "n", ltm_rag2_n03)
   result_df.to_csv("/home/yl3427/cylab/rag_tnm/rag_result/0929_ltm_rag2.csv", index=False)
