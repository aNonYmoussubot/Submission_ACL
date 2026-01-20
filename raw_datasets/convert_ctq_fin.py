import json
import os
import time
import re
import random 
import pandas as pd
from google import genai
from google.genai import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

API_KEY = "xxx" 
client = genai.Client(api_key=API_KEY)

MODEL_ID =  "gpt-5-thinking"

MAX_WORKERS = 1  

def json_table_to_markdown(table_data):
    try:
        header = table_data.get("header", [])
        rows = table_data.get("rows", [])

        if not header and not rows:
            return ""

        max_cols = len(header)
        if rows:
            row_lengths = [len(r) for r in rows]
            if row_lengths:
                max_cols = max(max_cols, max(row_lengths))

        if len(header) < max_cols:
            header += [""] * (max_cols - len(header))
            
        normalized_rows = []
        for row in rows:
            clean_row = list(row) 
            if len(clean_row) < max_cols:
                clean_row += [""] * (max_cols - len(clean_row))
            normalized_rows.append(clean_row)

        df = pd.DataFrame(normalized_rows, columns=header)
        
        return df.to_markdown(index=False, numalign="left", stralign="left")
    except Exception as e:
        return str(table_data)

def clean_json_text(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?", "", text)
        text = re.sub(r"```$", "", text)
    return text.strip()


SYSTEM_PROMPT = """
You are an expert data annotator specializing in Table Question Answering (TableQA). 
Your task is to convert a "Fact Verification" pair (Table + Claim) into a high-quality "Question-Answer" pair.

### GOAL
Generate a question that requires **reasoning** over the table to verify the claim, and provide the correct answer derived strictly from the table.

### REASONING TYPES TO IMPLEMENT
Analyze the Claim and generate a question fitting one of these categories:
1. **Comparison**: (e.g., "Which group has a higher value?", "Is A greater than B?")
2. **Arithmetic**: (e.g., "What is the difference between X and Y?", "What is the total of...")
3. **Superlative**: (e.g., "Which factor is ranked first?", "What is the lowest value in column X?")
4. **Aggregation**: (e.g., "How many items have value > 0.5?", "What is the average of...")
5. **Multiple Reasoning**: (e.g., Combining difference and ranking, "What is the difference between the highest and lowest value?")

### CRITICAL RULES
1. **Handling REFUTED Labels**:
   - The claim contains a WRONG number or logic.
   - **Do NOT ask "Is this claim true?"** (Avoid Yes/No questions).
   - **Do NOT repeat the wrong number** in the question.
   - **Question**: Ask for the *correct* value or relationship.
   - **Answer**: Must be the *correct* value from the table.
   - *Example*: Claim "A is 50" (False, Table says 30). -> QA: "What is the value of A?" -> "30".

2. **Handling SUPPORT Labels**:
   - The claim is true.
   - Identify the logic (Math/Comparison) used in the claim and formulate a question that leads to that result.

3. **Answer Format**:
   - Must be a **list of strings**.
   - Keep answers concise (numbers, entities).
   - Preserve special symbols (e.g., "0.40**") unless the question specifically asks to ignore them.

### OUTPUT JSON FORMAT
{
  "generated_question": "The natural language question",
  "generated_answers": ["The correct answer"],
  "reasoning_type": "The type used (e.g., Arithmetic)"
}
"""


def generate_qa(table_md, caption, claim, label):
    user_prompt = f"""
    ### Table Caption
    {caption}

    ### Table Content
    {table_md}

    ### Original Claim
    "{claim}"
    
    ### Label
    {label}

    Identify the logic in the claim and generate the corresponding Question and Answer.
    """

    max_retries = 5
    base_delay = 5  

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.1, 
                    response_mime_type="application/json"
                )
            )
            return json.loads(clean_json_text(response.text))
        
        except Exception as e:
            error_msg = str(e)
            retry_keywords = [
                "429",              
                "500", "503",       
                "SSL", "EOF",      
                "disconnected",     
                "closed connection" 
            ]

            is_retriable = any(k in error_msg for k in retry_keywords)

            if is_retriable and attempt < max_retries - 1:

                sleep_time = (base_delay * (2 ** attempt)) + random.uniform(0.5, 2.0)
                tqdm.write(f"⚠️ 警告: 网络或限流 ({error_msg[:30]}...). 重试 {attempt+1}/{max_retries}, 等待 {sleep_time:.1f}s")
                time.sleep(sleep_time)
            elif not is_retriable:
                tqdm.write(f"❌ API 错误 (不可重试): {error_msg}")
                return None
            else:
                tqdm.write(f"❌ 最终失败 (达到最大重试次数): {error_msg}")
                return None
    
    return None

def process_single_item(args):
    idx, item = args
    
    original_claim = item.get("question") or item.get("claim")
    if not original_claim:
        return idx, None

    label = "support"
    answers_list = item.get("answers", [])
    if isinstance(answers_list, list) and answers_list:
        label_text = str(answers_list[0]).lower()
        if "refut" in label_text:
            label = "refuted"
        elif "support" in label_text:
            label = "support"
    
    table_data = item.get("table", {})
    if not isinstance(table_data, dict):
        return idx, None
        
    caption = table_data.get("caption", "")
    table_md = json_table_to_markdown(table_data)

    qa_result = generate_qa(table_md, caption, original_claim, label)

    if qa_result:
        new_item = item.copy()
        
        new_item["original_claim"] = original_claim
        new_item["original_label"] = label
        
        new_item["question"] = qa_result.get("generated_question")
        new_item["answers"] = qa_result.get("generated_answers", [])
        new_item["reasoning_type"] = qa_result.get("reasoning_type")
        
        return idx, new_item
    else:
        return idx, None

def process_dataset_multithreaded(input_data_path):
    input_data = load_data(input_data_path)
    
    if not input_data:
        print("Dataset is empty or not found.")
        return []

    print(f"🚀 开始处理 {len(input_data)} 条数据，并发线程数: {MAX_WORKERS} ...")
    
    results_buffer = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_single_item, (i, item)) for i, item in enumerate(input_data)]
        
        for future in tqdm(as_completed(futures), total=len(input_data), desc="Processing", unit="it"):
            try:
                idx, result = future.result()
                if result:
                    results_buffer.append((idx, result))
            except Exception as e:
                tqdm.write(f"Worker Exception: {e}")

    results_buffer.sort(key=lambda x: x[0])
    return [res[1] for res in results_buffer]

def load_data(filepath):
    data = []
    if not os.path.exists(filepath):
        print(f"Error: 文件不存在 {filepath}")
        return data
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if not content: return []
        
        if content.startswith('['):
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                print("❌ JSON 解析失败，尝试逐行读取...")
        else:
            # 尝试 JSONL
            for line in content.split('\n'):
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except: pass
    return data

if __name__ == "__main__":

    raw_path = 'raw_datasets/fin/fin_final.json'  
    output_path = 'raw_datasets/fin/fin_qa.json'
    
    start_time = time.time()
    final_data = process_dataset_multithreaded(raw_path)
    end_time = time.time()
    
    if final_data:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
            
        print(f"\n✅ 处理完成！耗时 {end_time - start_time:.2f} 秒")
        print(f"✅ 有效数据: {len(final_data)} 条")
        print(f"✅ 文件已保存至: {output_path}")
    else:
        print("\n⚠️ 未生成任何有效数据，请检查输入文件或 API 配额。")