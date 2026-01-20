import json
import re
import os
import asyncio
import random
from typing import Dict, Any
from tqdm.asyncio import tqdm_asyncio

from openai import AsyncOpenAI
import openai


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_KEY_HERE")

MODEL_ID = "gpt-5-thinking"

INPUT_FILE = "./raw_datasets/wtq/test.json"
OUTPUT_FILE = "./processed_data/wtq_qa.json"

CONCURRENCY_LIMIT = 1          
API_REQUEST_CONCURRENCY = 1    
SLEEP_DELAY = 30              
ENABLE_JITTER = True         

API_TIMEOUT_S = 90.0
MAX_RETRIES = 5
BASE_BACKOFF_S = 2.0

client = AsyncOpenAI(api_key=OPENAI_API_KEY)



def table_to_markdown(table_obj):
    try:
        if isinstance(table_obj, str):
            return table_obj

        headers = table_obj.get("header", [])
        rows = table_obj.get("rows", [])

        if not headers:
            return "[Empty Table]"

        md_lines = []
        md_lines.append("| " + " | ".join(headers) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for row in rows:
            clean_row = [str(cell).replace("\n", " ") for cell in row]
            md_lines.append("| " + " | ".join(clean_row) + " |")

        return "\n".join(md_lines)
    except Exception as e:
        return f"[Table conversion error: {str(e)}]"


def extract_json_from_text(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {"error": "EMPTY_OUTPUT"}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pattern = r"```json\s*(\{.*?\})\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass

        pattern_raw = r"(\{.*\})"
        match_raw = re.search(pattern_raw, text, re.DOTALL)
        if match_raw:
            try:
                return json.loads(match_raw.group(1))
            except Exception:
                pass

        return {"error": "JSON_PARSE_FAILED", "raw_content": text}


TYPE1_FORMAT = {
    "type": "json_schema",
    "name": "wtq_type1_correct",
    "schema": {
        "type": "object",
        "properties": {
            "chain_of_thought": {"type": "string"},
            "answer": {"type": "string"},
        },
        "required": ["chain_of_thought", "answer"],
        "additionalProperties": False,
    },
    "strict": True,
}

TYPE2_FORMAT = {
    "type": "json_schema",
    "name": "wtq_type2_flawed",
    "schema": {
        "type": "object",
        "properties": {
            "flawed_chain_of_thought": {"type": "string"},
            "answer": {"type": "string"},
        },
        "required": ["flawed_chain_of_thought", "answer"],
        "additionalProperties": False,
    },
    "strict": True,
}

TYPE3_FORMAT = {
    "type": "json_schema",
    "name": "wtq_type3_wrong",
    "schema": {
        "type": "object",
        "properties": {
            "incorrect_chain_of_thought": {"type": "string"},
            "incorrect_answer": {"type": "string"},
        },
        "required": ["incorrect_chain_of_thought", "incorrect_answer"],
        "additionalProperties": False,
    },
    "strict": True,
}

TYPE4_FORMAT = {
    "type": "json_schema",
    "name": "wtq_type4_calc_error",
    "schema": {
        "type": "object",
        "properties": {
            "correct_logic_wrong_math_cot": {"type": "string"},
            "incorrect_answer": {"type": "string"},
        },
        "required": ["correct_logic_wrong_math_cot", "incorrect_answer"],
        "additionalProperties": False,
    },
    "strict": True,
}


class WTQFullGenerator:
    def __init__(self, request_sem: asyncio.Semaphore):
        self.request_sem = request_sem

    async def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
        fmt: Dict[str, Any],
        effort: str = "medium",
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """
        GPT-5 Thinking -> Responses API
        effort: minimal / low / medium / high
        """
        for attempt in range(MAX_RETRIES):
            try:
                async with self.request_sem:
                    resp = await asyncio.wait_for(
                        client.responses.create(
                            model=MODEL_ID,
                            instructions=system_prompt,
                            input=user_prompt,
                            temperature=temperature,
                            reasoning={"effort": effort},
                            text={"format": fmt},
                            store=False,
                        ),
                        timeout=API_TIMEOUT_S,
                    )

                txt = (resp.output_text or "").strip()
                if not txt:
                    return {"error": "EMPTY_OUTPUT"}

                try:
                    return json.loads(txt)
                except Exception:
                    return extract_json_from_text(txt)

            except asyncio.TimeoutError:
                return {"error": "TIMEOUT"}

            except Exception as e:
                status_code = getattr(e, "status_code", None)
                retriable = (
                    status_code in (429, 500, 503)
                    or isinstance(
                        e,
                        (
                            openai.RateLimitError,
                            openai.APIConnectionError,
                            openai.APITimeoutError,
                            openai.InternalServerError,
                        ),
                    )
                )

                if retriable and attempt < MAX_RETRIES - 1:
                    backoff = BASE_BACKOFF_S * (2 ** attempt)
                    if ENABLE_JITTER:
                        backoff *= random.uniform(0.8, 1.2)
                    await asyncio.sleep(backoff)
                    continue

                return {"error": str(e)}

        return {"error": "MAX_RETRIES_EXCEEDED"}

    async def generate_type_1_correct(self, table_str: str, question: str):
        system_prompt = (
            "You are an expert data analyst. Answer the question based on the table.\n\n"
            "IMPORTANT:\n"
            "- Provide a SHORT, high-level explanation (2-5 sentences). Do NOT provide hidden step-by-step chain-of-thought.\n"
            "- Ground your explanation in the table (mention rows/columns/entities used).\n\n"
            "Output valid JSON only."
        )

        user_prompt = f"""
Table:
{table_str}

Question: {question}

Output JSON format:
{{
  "chain_of_thought": "A short, high-level explanation grounded in the table (no hidden step-by-step).",
  "answer": "The final concise answer"
}}
""".strip()

        return await self._call_api(system_prompt, user_prompt, fmt=TYPE1_FORMAT, effort="high", temperature=0.2)

    async def generate_type_2_flawed(self, table_str: str, question: str, gold_answer: str, error_type="arithmetic"):
        strategies = {
            "arithmetic": "Make a subtle calculation slip, but still end with the target answer.",
            "grounding": "Cite a wrong cell/row confidently, but still end with the target answer.",
            "logic": "Use a plausible but incorrect formula/definition, but still end with the target answer.",
        }
        strategy_desc = strategies.get(error_type, strategies["arithmetic"])

        system_prompt = (
            "You are a confident but careless data analyst.\n"
            "Goal: Provide a flawed explanation that still concludes with the EXACT target answer.\n\n"
            "Rules:\n"
            "1) Never admit mistakes.\n"
            "2) Keep the tone confident and natural.\n"
            f"3) Strategy: {strategy_desc}\n\n"
            "Output valid JSON only."
        )

        user_prompt = f"""
Table:
{table_str}

Question: {question}
Target Answer: {gold_answer}

Output JSON format:
{{
  "flawed_chain_of_thought": "A plausible but flawed explanation with hidden {error_type} issues...",
  "answer": "{gold_answer}"
}}
""".strip()

        return await self._call_api(system_prompt, user_prompt, fmt=TYPE2_FORMAT, effort="medium", temperature=0.5)

    async def generate_type_3_wrong(self, table_str: str, question: str, gold_answer: str):
        system_prompt = (
            "You are a flawed reasoner.\n"
            "Generate a response where BOTH the explanation and the final answer are incorrect.\n"
            "1) Misread the table or pick the wrong column.\n"
            "2) Ensure the final answer is different from the hidden truth.\n"
            "3) Keep it plausible.\n\n"
            "Output valid JSON only."
        )

        user_prompt = f"""
Table:
{table_str}

Question: {question}
(Hidden Truth: "{gold_answer}" - DO NOT OUTPUT THIS)

Output JSON format:
{{
  "incorrect_chain_of_thought": "A plausible but wrong explanation...",
  "incorrect_answer": "A wrong final answer"
}}
""".strip()

        return await self._call_api(system_prompt, user_prompt, fmt=TYPE3_FORMAT, effort="low", temperature=0.7)

    async def generate_type_4_calc_error(self, table_str: str, question: str, gold_answer: str):
        system_prompt = (
            "You are a data analyst with correct table understanding but poor arithmetic.\n"
            "Requirements:\n"
            "1) Correctly identify the relevant rows/columns/numbers.\n"
            "2) Set up the correct operation.\n"
            "3) Make the FINAL arithmetic result wrong so the final answer differs from the hidden truth.\n\n"
            "Output valid JSON only."
        )

        user_prompt = f"""
Table:
{table_str}

Question: {question}
(Hidden Truth: "{gold_answer}" - DO NOT OUTPUT THIS AS THE FINAL ANSWER)

Output JSON format:
{{
  "correct_logic_wrong_math_cot": "Correctly cite numbers and formula, but describe a wrong calculation result.",
  "incorrect_answer": "The resulting wrong answer"
}}
""".strip()

        return await self._call_api(system_prompt, user_prompt, fmt=TYPE4_FORMAT, effort="medium", temperature=0.6)

async def process_item(item_sem, generator: WTQFullGenerator, item: Dict[str, Any]):
    async with item_sem:
        try:
            q_id = item.get("id", "unknown")
            question = item.get("question", "")

            answers = item.get("answers", [])
            if isinstance(answers, list) and len(answers) > 0:
                gold_answer = str(" ".join(map(str, answers)))
            else:
                gold_answer = str(answers) if answers else "Unknown"

            table_data = item.get("table", {})
            if not isinstance(table_data, dict):
                return None

            table_md = table_to_markdown(table_data)

            
            t1, t2_g, t2_a, t2_l, t3, t4 = await asyncio.gather(
                generator.generate_type_1_correct(table_md, question),
                generator.generate_type_2_flawed(table_md, question, gold_answer, "grounding"),
                generator.generate_type_2_flawed(table_md, question, gold_answer, "arithmetic"),
                generator.generate_type_2_flawed(table_md, question, gold_answer, "logic"),
                generator.generate_type_3_wrong(table_md, question, gold_answer),
                generator.generate_type_4_calc_error(table_md, question, gold_answer),
            )

            result = {
                "id": q_id,
                "original_question": question,
                "gold_answer": gold_answer,
                "table_content": table_data,
                "table_md": table_md,
                "generated_samples": {
                    "type1_correct": t1,
                    "type2_grounding_error": t2_g,
                    "type2_arithmetic_error": t2_a,
                    "type2_logic_error": t2_l,
                    "type3_fully_wrong": t3,
                    "type4_calc_error": t4,
                },
            }

            if SLEEP_DELAY > 0:
                actual_sleep = float(SLEEP_DELAY)
                if ENABLE_JITTER:
                    actual_sleep *= random.uniform(0.8, 1.2)
                await asyncio.sleep(actual_sleep)

            return result

        except Exception as e:
            print(f"Error processing ID {item.get('id')}: {e}")
            return None


async def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)[171:211]

    print(
        f"Loaded {len(data)} items. "
        f"item_concurrency={CONCURRENCY_LIMIT}, api_concurrency={API_REQUEST_CONCURRENCY}, "
        f"sleep={SLEEP_DELAY}s..."
    )

    item_sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    request_sem = asyncio.Semaphore(API_REQUEST_CONCURRENCY)

    generator = WTQFullGenerator(request_sem=request_sem)

    tasks = [process_item(item_sem, generator, item) for item in data]
    results = []

    for fut in tqdm_asyncio.as_completed(tasks, desc="Generating All Types"):
        res = await fut
        if res:
            results.append(res)

    print(f"Saving {len(results)} results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Done!")


if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
