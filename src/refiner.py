from typing import Optional, List
import pandas as pd
from src.schema import CoTTrace
from src.pipeline import TrustTablePipeline
from src.llm_engine import LLMEngine
from utils.logger import setup_logger

logger = setup_logger("BlindRefiner")

class BlindIterativeRefiner:
    def __init__(self, table_df: pd.DataFrame, llm: LLMEngine, refinement_enabled: bool = True):

        self.table_df = table_df
        self.llm = llm
        self.pipeline = TrustTablePipeline(table_df)
        self.refinement_enabled = refinement_enabled 

    def solve(self, question: str, max_retries: int = 3, refinement_enabled: Optional[bool] = None) -> dict:

        do_refine = self.refinement_enabled if refinement_enabled is None else refinement_enabled
        
        effective_max_retries = max_retries if do_refine else 0
        
        history = []
        
        logger.info(f"Start solving: {question} | Refinement: {'ON' if do_refine else 'OFF'}")
        
        current_cot_text = self._generate_initial_cot(question)
        
        for attempt in range(effective_max_retries + 1):
            if attempt > 0:
                logger.info(f"=== Iteration {attempt} (Refinement) ===")
            else:
                logger.info(f"=== Iteration {attempt} (Initial) ===")
            
            # 1. Atomic Decomposition
            steps = self.llm.decompose_cot(current_cot_text)
            current_trace = CoTTrace(question=question, steps=steps, raw_text=current_cot_text)
            
            # 2. Logic Auditing (Verification)
            is_valid, error_report = self.pipeline.run(current_trace)
            
            history.append({
                "iteration": attempt,
                "cot": current_cot_text,
                "valid": is_valid,
                "error": error_report
            })
            
            if is_valid:
                logger.info("Verification PASSED. Outputting result.")
                return {
                    "final_answer": self._extract_answer(current_cot_text),
                    "trace": current_trace,
                    "status": "Verified",
                    "history": history
                }

            if attempt < effective_max_retries:
                logger.warning(f"Verification FAILED. Triggering Refinement. Reason: {error_report.get('reason', 'Unknown')}")
                
                # 3. Iterative Refinement (Self-Correction)
                current_cot_text = self._refine_cot(question, current_cot_text, error_report)
            else:
                if not do_refine:
                    logger.info("Verification FAILED. Refinement is DISABLED. Returning initial result.")
                    status = "VerificationFailed"
                else:
                    logger.error("Max retries reached. Returning best effort.")
                    status = "MaxRetriesReached" 

                return {
                    "final_answer": self._extract_answer(current_cot_text),
                    "trace": current_trace,
                    "status": status,
                    "history": history
                }

    def _refine_cot(self, question: str, old_cot: str, error: dict) -> str:

        module = error.get("module")
        
        if module == "FactChecker":
            return self._refine_grounding(question, old_cot, error)
            
        elif module == "Z3Auditor":
            logger.info("Delegating to Logic Auditor for Proof Refinement...")
            return self.llm.refine_logic_proof(question, old_cot, error)
        
        else:
            return old_cot

    def _refine_grounding(self, question: str, old_cot: str, error: dict) -> str:
        bad_step = error.get("step_content", "")
        reason = error.get("reason", "")
        
        system_prompt = """You are a Refinement Agent checking data against a table.
A previous reasoning chain contained a HALLUCINATION (Data Grounding Error).
Your goal is to rewrite the reasoning to strictly adhere to the table content.
"""
        table_snippet = self.table_df.to_string() 

        user_prompt = f"""
### Table Data
{table_snippet}

### Question
{question}

### Failed Attempt
{old_cot}

### Error Feedback (From FactChecker)
- The step "{bad_step}" is invalid.
- Reason: {reason}

### Instruction
Rewrite the Chain-of-Thought. 
1. CORRECT the specific data error identified above.
2. Ensure all other steps are also supported by the table.
3. Keep the reasoning concise.

### Corrected Chain-of-Thought:
"""
        response = self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()

    def _generate_initial_cot(self, question: str) -> str:
        table_str = self.table_df.to_string()
        prompt = f"Table:\n{table_str}\n\nQuestion: {question}\n\nAnswer step-by-step:"
        response = self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()

    def _extract_answer(self, cot: str) -> str:

        lines = cot.split('\n')

        for line in reversed(lines):
            if line.strip():
                return line.replace("Answer:", "").replace("answer is", "").strip()
        return ""