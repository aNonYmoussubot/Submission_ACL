import json
import re
from openai import OpenAI
from configs.config import Config
from utils.logger import setup_logger
from typing import List, Dict
logger = setup_logger("LLMEngine")

class LLMEngine:
    def __init__(self):
        self.client = OpenAI(api_key=Config.API_KEY, base_url=Config.BASE_URL)
        self.model = Config.MODEL_NAME

    def autoformalize_to_z3(self, premise_text: str, conclusion_text: str, table_context: str = "") -> str:
        
        system_prompt = """You are an expert in Formal Verification.
Your task is to verify if a Conclusion follows from the Premise, GIVEN the Table Data context.

### STRATEGY: Data-Augmented Verification
1. **The Closed World Assumption**: 
   - The provided "Table Context" contains the ACTUAL values from the real world. 
   - Treat these values as hard constraints (Axioms).
   
2. **Handling "Max/Min/Rank"**:
   - If the conclusion claims "X is the highest", DO NOT just look at the premise. 
   - Compare X against the list of values provided in the Table Context.
   - Logic: `If X >= max(Table_Values), return True`.

3. **Output**:
   - Write a `solve_logic()` function returning `(bool, model)`.
   - Use `z3.If` for logic, or simple Python assertions if checking against fixed data lists.
"""

        user_prompt = f"""
### Table Context (Ground Truth)
{table_context}

### Premise
"{premise_text}"

### Conclusion
"{conclusion_text}"

### Task
Write Python Z3 code to verify the conclusion.
### Example Template
```python
def solve_logic():
    s = Solver()
    # 1. Variables
    A_Total, A_Gold = Ints('A_Total A_Gold')
    
    # 2. Premise Constraints
    s.add(A_Total == 19)
    
    # 3. The Logic to Verify: "Total > 10 implies Gold > 5"
    conclusion = Implies(A_Total > 10, A_Gold > 5)
    
    # 4. Proof by Contradiction (Find Counter-example)
    s.add(Not(conclusion))
    
    if s.check() == sat:
        return False, s.model() # Invalid
    return True, None # Valid
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1 
            )
            
            raw_content = response.choices[0].message.content
            return self._clean_code(raw_content)

        except Exception as e:
            logger.error(f"LLM Generation Failed: {e}")
            return "def solve_logic(): raise Exception('LLM Generation Failed')"
        
        
    def autoformalize_to_z3_1(self, premise_text: str, conclusion_text: str) -> str:
        
        system_prompt = """You are an expert in Formal Verification and Z3 Theorem Prover.
Your task is to translate Natural Language Reasoning into executable Python Z3 code to verify its logical validity.
"""

        user_prompt = f"""
### Context
I have a TableQA reasoning step that needs verification.
- **Premise (Context)**: "{premise_text}"
- **Conclusion (Step to Verify)**: "{conclusion_text}"

### Task
Write a Python script using `z3` to verify if the Conclusion follows from the Premise.
1. **Detect Rule Definitions**:
   - If the Conclusion is strictly **defining a rule** (e.g., "Smaller time is better", "A win gives 3 points"), this is a **DEFINITION**, not a deduction.
   - **Action**: Return `True` immediately. Do not try to prove a definition.
   - **Code Pattern**:
     ```python
     def solve_logic():
         return True, None # It's a definition/axiom
     ```
### CRITICAL STRATEGY: "Implicit Axioms"
2. **Detect Implicit Definitions**: 
   - If the reasoning relies on **Sequential Order** (e.g., "5 comes after 4", "Item N+1 follows Item N"), treat this as a **TRUE AXIOM**.
   - If the reasoning relies on **Table Structure** (e.g., "The row below row 1 is row 2"), treat this as a **TRUE AXIOM**.
   
3. **Where to add constraints**:
   - Add these axiomatic definitions to `s.add(...)` BEFORE the contradiction check.
   - **Example**: If text says "Since it's sequential, GL-B-6 follows GL-B-5":
     - `s.add(GL_B_6 == GL_B_5 + 1)`  <-- Add this as a FACT.
     - Then check for contradiction.

4. **Detect Logic Flaws (Spurious Correlations)**:
   - If the reasoning uses arbitrary math (e.g., "Sum of hyphens equals ID"), **DO NOT** add this as a fact. Treat it as the hypothesis to test.
   - **Example**: "ID is sum of hyphens":
     - `s.add(Hyphen_Count == 6)`
     - `conclusion = (ID == Hyphen_Count)`
     - `s.add(Not(conclusion))` -> This will likely be SAT (Invalid), which is what we want.

### Implementation Requirements
1. Define function `def solve_logic():` returning `(bool, model)`. (True=Valid, False=Invalid).
2. Use `Ints` or `Reals` for abstract values.
3. **Counter-Example Check**: Use `s.add(Not(conclusion))` to find invalid cases.

### Output
Return ONLY the python code block.

### Example Template
```python
def solve_logic():
    s = Solver()
    # 1. Variables
    A_Total, A_Gold = Ints('A_Total A_Gold')
    
    # 2. Premise Constraints
    s.add(A_Total == 19)
    
    # 3. The Logic to Verify: "Total > 10 implies Gold > 5"
    conclusion = Implies(A_Total > 10, A_Gold > 5)
    
    # 4. Proof by Contradiction (Find Counter-example)
    s.add(Not(conclusion))
    
    if s.check() == sat:
        return False, s.model() # Invalid
    return True, None # Valid
```"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1 
            )
            
            raw_content = response.choices[0].message.content
            return self._clean_code(raw_content)

        except Exception as e:
            logger.error(f"LLM Generation Failed: {e}")
            return "def solve_logic(): raise Exception('LLM Generation Failed')"
    def _clean_code(self, text: str) -> str:

        pattern = r"```(?:python)?\s*(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        return text.strip()
    

    def decompose_cot(self, cot_text: str) -> List[Dict]:
        
        system_prompt = """You are a Reasoning Parser for TableQA tasks.
Your goal is to break down a raw Chain-of-Thought (CoT) paragraph into atomic, executable steps.

For each step, assign a **Type**:
1. **fact**: The step involves looking up specific data, rows, or values in the table (e.g., "The gold medals for Brazil is 7", "Locate the row for GL-B-5").
2. **inference**: The step involves calculation, comparison, logical deduction, or applying rules (e.g., "Since 19 > 10", "The next item in the sequence is...").

### Output Format
Return a JSON object with a key "steps", containing a list of objects.
Example:
{
  "steps": [
    {"content": "Look at the row for Brazil,Brazil has 19 total medals.", "type": "fact"},
    {"content": "Since 19 is greater than 10, Brazil wins.", "type": "inference"}
  ]
}
"""

        user_prompt = f"""
### Raw CoT Text
"{cot_text}"

### Task
Decompose this text into atomic steps. Return JSON.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}, 
                temperature=0.0 
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("steps", [])

        except Exception as e:
            logger.error(f"CoT Decomposition Failed: {e}")
            return [{"content": cot_text, "type": "inference"}]
        
    def generate_pandas_check(self, claim: str, columns: list, sample_data: str) -> str:
        system_prompt = """You are a Python Pandas Expert for TableQA verification.
Your goal is to write a Python function `verify_fact(df)` that checks if a natural language claim is supported by the given DataFrame.

### ROBUSTNESS RULES (CRITICAL)
1. **Fuzzy String Matching**: Tables often contain hidden spaces or newlines. 
   - ALWAYS use `.str.strip()` and `.astype(str)` for comparisons.
   - Example: `df[df['Yacht'].str.contains('Ausmaid', na=False)]`
2. **Partial Match**: If a specific time '3:06:02:29' is mentioned, try to match the most unique part (like the yacht name) first, then verify the time value in that row.
3. **Column Names**: Use the exact column names provided in the Schema.

### Requirements
1. **Function Signature**: `def verify_fact(df): -> bool`
2. **Logic**:
   - Return `True` if the data in `df` strictly supports the claim.
   - Return `False` if the data contradicts the claim or the entity is not found.
3. **Robustness**:
   - Use Modern Pandas: Use `df.map()` instead of `df.applymap()`.
   - If the claim is about **intent** (e.g., "We need to check column X", "Let's examine the order"), it is NOT a falsifiable fact.
   - In these cases, simply return `True`.
   - **Do NOT** try to verify "order" or "sequence" unless specific row numbers are mentioned.
   - Handle string matching carefully (strip spaces, ignore case if needed).
   - Convert data types if necessary (e.g., `df['Score'].astype(str)`).
   - Use `.values[0]` carefully; check if the filtered dataframe is empty first.
4. **Output**: Return ONLY the code block.
"""

        user_prompt = f"""
### Table Schema
- Columns: {columns}
- Sample Data (First row): {sample_data}

### Claim to Verify
"{claim}"

### Task
Write the `verify_fact(df)` function.
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            return self._clean_code(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Pandas Gen Failed: {e}")
            return "def verify_fact(df): return False"
        

    def refine_logic_proof(self, question: str, old_cot: str, error_report: dict) -> str:
        module = error_report.get("module", "")
        reason = error_report.get("reason", "")
        failed_step = error_report.get("step_content", "")

        system_prompt = """You are a Formal Logic Auditor. Your role is to evaluate and fortify a reasoning chain (CoT) that failed a symbolic verifier (Z3/Pandas).

### AUDIT PHILOSOPHY:
- **Case A: Logic Leak (Incomplete Proof)**: The reasoning is correct but "leaky". For example, saying "7 is the max" without proving others are smaller. 
- *Refinement*: You must explicitly cite the values of ALL competitors from the table to "close the logical world".
- **Case B: Spurious Logic (Wrong Rule)**: The reasoning uses a rule that isn't supported by the table (e.g., "Sail number determines speed").
- *Refinement*: You must acknowledge the error and switch to a standard lookup or sequential logic.
- **Case C: Hallucination**: The reasoning cited data that isn't in the table.
- *Refinement*: Re-check the table and use grounded facts.

### OUTPUT:
Provide a refined, step-by-step Chain-of-Thought that is robust enough to be logically irrefutable.
"""

        user_prompt = f"""
### Original Question
"{question}"

### Failed Reasoning Trace
"{old_cot}"

### Verifier Feedback
- **Failed Module**: {module}
- **Faulty Step**: "{failed_step}"
- **Technical Objection**: {reason}

### Audit Instruction
Analyze the Technical Objection. If a counter-example was found, the logic is "leaky". 
Rewrite the Reasoning Chain to be a "Strict Proof". If it's a comparison task, you MUST explicitly enumerate the values of the other candidates to block the solver from finding counter-examples.

### Fortified Reasoning Chain:
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                top_p=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Refinement Failed: {e}")
            return old_cot