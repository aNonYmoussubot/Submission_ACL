import json
import os
import asyncio
import functools
import pandas as pd
import io
from tqdm.asyncio import tqdm_asyncio
from src.llm_engine import LLMEngine
from utils.logger import setup_logger

import pandas as pd
import numpy as np

logger = setup_logger("Code_Verifier")

class CodeBasedVerifier:
    def __init__(self):
        self.llm = LLMEngine()

        self.model_name = self.llm.model 
        self.temperature = 0.0

    def construct_code_gen_prompt(self, table_str, question, reasoning, answer):

        system_prompt = """You are a Computational Logic Auditor.
Your goal is to verify a "Reasoning Trace" by converting it into an executable Python verification script using Pandas.

### INSTRUCTIONS
1. **Decompose**: First, break down the reasoning trace into atomic verification steps in comments.
2. **Implement**: Write a Python function `def verify_reasoning(df):` that checks each step against the DataFrame `df`.
3. **Assert**: 
   - Verify specific data claims (e.g., `assert df.iloc[0]['Year'] == '2005'`).
   - Verify calculations (e.g., `calculated_sum = df['Points'].sum(); assert calculated_sum == 10`).
   - Verify the final answer consistency.
4. **Return**: The function must return `True` ONLY if all checks pass. If any check fails or data is missing, return `False`.

### ROBUSTNESS RULES
- The dataframe `df` contains strings. You MUST convert columns to numeric types (e.g., `pd.to_numeric`) before doing math.
- Handle formatting (e.g., remove ',' or '$') before conversion.
- Use strict assertions.

### Output Format
Return ONLY the python code block containing the function.
```python
def verify_reasoning(df):
    # Step 1: ...
    # Code ...
    return True
```
"""
        user_prompt = f"""
Table Schema & Data Snippet
{table_str}

Question
{question}

Candidate Reasoning to Verify
"{reasoning}"

Predicted Answer
"{answer}"

Task
Generate the Python verification code. """
        return system_prompt, user_prompt
    def execute_verification_code(self, code_str, table_content):

        try:
            
            if isinstance(table_content, dict) and "header" in table_content and "rows" in table_content:
                df = pd.DataFrame(table_content["rows"], columns=table_content["header"])
            else:

                return "ERROR_DATA_FORMAT", "Missing structured table content"


            code_str = code_str.replace("```python", "").replace("```", "").strip()
            

            local_scope = {}
            global_scope = {"pd": pd, "np": np}
            

            exec(code_str, global_scope, local_scope)
            

            if "verify_reasoning" not in local_scope:
                return "ERROR_NO_FUNCTION", "Function 'verify_reasoning' not found in generated code"
            
            verify_func = local_scope["verify_reasoning"]
            

            result = verify_func(df)
            
            if result is True:
                return "ACCEPT", "Verification passed execution."
            else:
                return "REJECT", "Verification function returned False."

        except AssertionError as e:
            return "REJECT", f"Assertion Failed: {str(e)}"
        except Exception as e:
            return "REJECT", f"Execution Error: {str(e)}"

    async def verify_one_sample(self, original_item, sample_type, specific_subtype, sample_data):
        """
        验证单个样本
        """
        try:

            if not isinstance(sample_data, dict) or "error" in sample_data:
                return None


            table_content = original_item.get("table_content", {})

            table_str = original_item.get("table_md", str(table_content))
            
            question = original_item.get("original_question", "")


            reasoning = ""
            answer = ""
            

            if "chain_of_thought" in sample_data: reasoning = sample_data["chain_of_thought"]
            elif "flawed_chain_of_thought" in sample_data: reasoning = sample_data["flawed_chain_of_thought"]
            elif "correct_logic_wrong_math_cot" in sample_data: reasoning = sample_data["correct_logic_wrong_math_cot"]
            elif "incorrect_chain_of_thought" in sample_data: reasoning = sample_data["incorrect_chain_of_thought"]
            
            if "answer" in sample_data: answer = sample_data["answer"]
            elif "incorrect_answer" in sample_data: answer = sample_data["incorrect_answer"]
            elif "pred_answer" in sample_data: answer = sample_data["pred_answer"]

            if not reasoning or not answer:
                return None

            sys_p, user_p = self.construct_code_gen_prompt(table_str, question, reasoning, answer)
            
            api_call_func = functools.partial(
                self.llm.client.chat.completions.create,
                model=self.model_name,
                messages=[
                    {"role": "system", "content": sys_p},
                    {"role": "user", "content": user_p}
                ],
                temperature=self.temperature,
                timeout=60.0 
            )

            response = await asyncio.wait_for(
                asyncio.to_thread(api_call_func),
                timeout=70.0 
            )
            
            generated_code = response.choices[0].message.content


            decision, rationale = self.execute_verification_code(generated_code, table_content)
            
            final_decision = "ACCEPT" if decision == "ACCEPT" else "REJECT"

            return {
                "id": original_item.get("id"),
                "target_type": sample_type,
                "specific_subtype": specific_subtype,
                "verifier_decision": final_decision,
                "verifier_rationale": f"Decision: {decision}\nMsg: {rationale}\n\nCode:\n{generated_code}"
            }

        except asyncio.TimeoutError:
            logger.warning(f"TIMEOUT: Verification timed out for {original_item.get('id')}")
            return None
        except Exception as e:
            logger.error(f"Verification process failed: {e}")
            return None
        


async def main(): 

    INPUT_FILE = "./processed_data/wtq_qa_small.json" 
    OUTPUT_FILE = "./output/deepseek/wtq_pot_verifier_results.json" 
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found.")
        return

    print(f"Loading data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)



    verifier = CodeBasedVerifier()

    key_mapping = {
        "type1_correct": "type1_golden",
        "type2_grounding_error": "type2_spurious",
        "type2_arithmetic_error": "type2_spurious",
        "type2_logic_error": "type2_spurious",
        "type3_fully_wrong": "type3_wrong",
        "type4_calc_error": "type4_calc_error"
    }

    tasks = []

    for item in data:
        gen_samples = item.get("generated_samples", {})
        if not gen_samples: continue

        for json_key, std_type in key_mapping.items():
            if json_key in gen_samples:
                sample_data = gen_samples[json_key]
                task = verifier.verify_one_sample(
                    original_item=item, 
                    sample_type=std_type,
                    specific_subtype=json_key,
                    sample_data=sample_data
                )
                tasks.append(task)

    print(f"Created {len(tasks)} verification tasks.")

    results = []

    sem = asyncio.Semaphore(10) 

    async def sem_task(t):
        async with sem:
            return await t

    async_tasks = [asyncio.create_task(sem_task(t)) for t in tasks]

    if not async_tasks:
        print("No tasks created.")
        return

    for f in tqdm_asyncio.as_completed(async_tasks, desc="Code-Based Verification"):
        res = await f
        if res:
            results.append(res)

    print(f"Saving {len(results)} results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print("Done.")

if __name__ == "__main__":

    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())