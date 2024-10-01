system_instruction = "You are an expert at interpreting pathology reports for cancer staging."
system_instruction_short_ans_t14 = "You are an expert at interpreting pathology reports for cancer staging. You provide only one word answers. You must answer from the following four options:  T1, T2, T3, T4.  Here is an example of a well formatted answer: 'Ti' where i is from 1 to 4. You do not output any additional text."

basic_prompt = """You are provided with a pathology report for a cancer patient. 
Please review this report and determine the pathologic stage of the patient's cancer. 

Here is the report:
```
{report}
```

Does this patient have metastatic disease? Please answer only Yes or No.
"""

few_shots_prompt = """You are provided with a pathology report for a cancer patient. 
Please review this report and determine the pathologic stage of the patient's cancer. 

Here are some excerpts:

Report:the patient has a T1N2M0 squamous cell. carcinoma of the right base of tongue, involving the left. glossotonsillar sulcus extending into the lingual tonsil. Gross Description: Received are eleven appropriately labeled containers.
Answer:No
 
Report:A. ADIPOSE TISSUE WITH METASTATIC RENAL CELL CARCINOMA, WITH EXTENSIVE SARCOMATOID. DIFFERENTIATION. B. ANGIOLYMPHATIC INVASION IS IDENTIFIED. C. NO PANCREATIC TISSUE IS IDENTIFIED. D. THE NEOPLASM EXTENDS TO THE RESECTION MARGINS. E. NO EVIDENCE OF NEOPLASIA IN 4 (0/4) LYMPH NODES.
Answer:Yes
 
Report:REGIONAL LYMPH NODES (N). No regional lymph node metastasis histologically, no examination for ITC (pNo). DISTANT METASTASIS (M). Separate tumor nodules in contralateral lobe; tumor with pleural nodules or malignant pleural or pericardial effusion. (M1a). STAGE GROUPING. The overall international stage is T2a/NO//M1a (Stage IV).
Answer:Yes
 
Report:PATIENT HISTORY: DATE of LMP: DATE OF LAST DELIVERY. PRE-OP DIAGNOSIS: STAGE IV RIGHT BREAST CANCER. POST-OP DIAGNOSIS: SAME. OPERATIVE PROCEDURE: MODIFIED RADICAL RIGHT MASTECTOMY. 
Answer:Yes
 
Report:ATHOLOGIC STAGING. EXTENT OF INVASION. pT3. (Tumor extends beyond the pancreas but without involvement of. the celiac axis or the superior mesenteric artery). REGIONAL LYMPH NODES. pN1. (Regional lymph node metastasis). Total nodes: 21. Total positive nodes: 11. DISTANT METASTASIS. pM0. (No distant metastasis). PATHOLOGIC STAGE SUMMARY. Final TNM: pT3N1M0. stage: IIB.
Answer:No
 
Report:Pathologic Staging (pTNM): TNM descriptors: Primary tumor (pT): pT4. Regional lymph nodes (pN): pNO. Number examined: 21. Number involved: 0. Distant metastasis (pM): pM0 (clinical assessment). Additional pathologic findings: Tumor erodes mandibular cortex but does not. invade medullary component. 
Answer:No

Here is the report:

Report:{report}

Does this patient have metastatic disease? Please answer only Yes or No.
"""

basic_prompt_t14 = """You are provided with a pathology report for a cancer patient.
Please review this report and determine the pathologic stage of the patient's cancer.

Here is the report:
```
{report}
```

What is the T stage from this report? Ignore any substaging information.  Please select from the following four options:  T1, T2, T3, T4.
Only provide the T stage with no additional text. Here is an example of a well formatted answer: "Ti" where i is from 1 to 4.
"""

dfs_prompt_t14 = """Using the following reports as a guide, your task is to review the provided pathology report for a cancer patient and determine the T stage. Ignore any substaging information.  You must select from the following four options:  T1, T2, T3, T4.
Only provide the T stage with no additional text.  
{demonstrations} 

Report:{report}
Answer: 
"""

basic_prompt_n03 = """You are provided with a pathology report for a cancer patient.
Please review this report and determine the pathologic stage of the patient's cancer.
 
Here is the report:
```
{report}
```

What is the N stage from this report? Please select from the following four options:  N0, N1, N2, N3
Only provide the N stage with no additional text. Here is an example of a well formatted answer: "Ni" where i is from 0 to 3.
"""

dfs_prompt_n03 = """Using the following reports as a guide, your task is to review the provided pathology report for a cancer patient and determine the N stage. Ignore any substaging information.  You must select from the following four options:  N0, N1, N2, N3.
Only provide the N stage with no additional text.  
{demonstrations} 

Report:{report}
Answer: 
"""