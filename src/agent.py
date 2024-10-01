import json
from typing import List, Dict, Union, Optional, Tuple
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from pydantic import BaseModel, Field

class Response(BaseModel):
    reasoning: str = Field(description="Step-by-step explanation of how you interpreted the report to determine the cancer stage.")
    stage: str = Field(description="The cancer stage determined from the report.")
 
class Agent:
    def __init__(self):
        self.client = OpenAI(
            api_key = "empty",
            base_url = "http://localhost:8000/v1",
            timeout=120.0
            )
        
        self.schema = Response.model_json_schema()

    def get_response(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.client.chat.completions.create(
                model = "mistralai/Mixtral-8x7B-Instruct-v0.1",
                messages=messages,
                extra_body={"guided_json":self.schema},
                temperature = 0.1
            )

            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        
    def get_tnm_stage(self, df, prompt_method, prompt, stage_type, context = ""):
        parsing_error = 0
        pbar = tqdm(total=df.shape[0])

        for idx, row in df.iterrows():
            report = row['text']
            if context:
                formatted_prompt = prompt.format(report=report, context=context)
            else:
                formatted_prompt = prompt.format(report=report)

            response = self.get_response(formatted_prompt)

            if response:
                df.at[idx, f'{prompt_method}_{stage_type}_reasoning'] = response['reasoning']
                df.at[idx, f'{prompt_method}_{stage_type}_stage'] = response['stage']
            else:
                parsing_error += 1
            pbar.update(1)
        pbar.close()
        print(f"Total parsing errors: {parsing_error}")
        return df

                
           
    


        
                    
    

