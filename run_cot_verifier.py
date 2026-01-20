import json
import os
import asyncio
import functools
from tqdm.asyncio import tqdm_asyncio
from src.llm_engine import LLMEngine
from utils.logger import setup_logger

logger = setup_logger("CoT_Verifier_FineTuned")

class StandardCoTVerifier:
    def __init__(self):
        self.llm = LLMEngine()
        self.model_name = self.llm.model 
        self.temperature = 0.0

    def construct_verification_prompt(self, table_str, question, reasoning, answer):
        """
        构建 'LLM-as-a-Judge' 的 CoT Prompt
        """
        system_prompt = """You are a strict Logic Auditor for TableQA tasks.
Your goal is to verify whether a given "Reasoning Trace" and "predicted Answer" are strictly correct based on the provided Table.

### Criteria for Judgment
1. **Grounding**: Does the reasoning cite specific, correct values from the table?
2. **Logic**: Is the calculation or deduction logically sound? (e.g., is the formula correct?)
3. **Consistency**: Does the predicted answer match the reasoning?

### Output Format
- First, provide a step-by-step analysis of the errors (if any).
- Finally, end with exactly "JUDGMENT: ACCEPT" or "JUDGMENT: REJECT".
"""
        
        user_prompt = f"""
### Table Context
{table_str}

### Question
{question}

### Candidate Solution to Verify
**Reasoning Trace:**
{reasoning}

**Predicted Answer:**
{answer}

### Task
Analyze the candidate solution step-by-step. Is it fully correct?
"""
        return system_prompt, user_prompt

    async def verify_one_sample(self, original_item, sample_type, specific_subtype, sample_data):
        """
        验证单个样本 (修复了变量名错误)
        """
        try:
            if not isinstance(sample_data, dict) or "error" in sample_data:
                return None

            table_str = original_item.get("table_md", "")
            if not table_str and "table_content" in original_item:
                 table_str = str(original_item["table_content"]) 
            
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

            sys_p, user_p = self.construct_verification_prompt(table_str, question, reasoning, answer)
            
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
            
            raw_content = response.choices[0].message.content

            decision = "UNKNOWN"
            if "JUDGMENT: ACCEPT" in raw_content:
                decision = "ACCEPT"
            elif "JUDGMENT: REJECT" in raw_content:
                decision = "REJECT"
            else:
                last_line = raw_content.strip().split('\n')[-1].upper()
                if "ACCEPT" in last_line: decision = "ACCEPT"
                elif "REJECT" in last_line: decision = "REJECT"

            return {
                "id": original_item.get("id"),
                "target_type": sample_type,
                "specific_subtype": specific_subtype,
                "verifier_decision": decision,
                "verifier_rationale": raw_content
            }

        except asyncio.TimeoutError:
            logger.warning(f"TIMEOUT: Verification timed out for {original_item.get('id')} - {specific_subtype}")
            return None
        except Exception as e:
            logger.error(f"Verification failed for {original_item.get('id')} - {specific_subtype}: {e}")
            return None

async def main():
    INPUT_FILE = "./processed_data/wtq_qa_small.json" 
    OUTPUT_FILE = "./output/deepseek/wtq_cot_verifier_results.json"
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found.")
        return

    print(f"Loading data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    

    verifier = StandardCoTVerifier()
    
    key_mapping = {
        "type1_correct": "type1_golden",
        "type2_grounding_error": "type2_spurious",
        "type2_arithmetic_error": "type2_spurious",
        "type2_logic_error": "type2_spurious",
        "type3_fully_wrong":"type3_fully_wrong",
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

    print(f"Created {len(tasks)} verification tasks from {len(data)} source items.")
    
    results = []

    sem = asyncio.Semaphore(20) 
    
    async def sem_task(t):
        async with sem:
            return await t

    async_tasks = [asyncio.create_task(sem_task(t)) for t in tasks]
    
    if not async_tasks:
        print("No tasks created.")
        return


    for f in tqdm_asyncio.as_completed(async_tasks, desc="Standard CoT Verifying"):
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