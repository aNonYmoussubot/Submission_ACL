from typing import Tuple, Optional, List
import pandas as pd
from src.schema import CoTTrace, VerificationResult, ReasoningStep
from src.verifiers.fact_checker import FactChecker
from src.verifiers.z3_auditor import Z3Auditor
from utils.logger import setup_logger

logger = setup_logger("TrustTablePipeline")

class TrustTablePipeline:
    def __init__(self, table_df: pd.DataFrame):
        self.table = table_df
        self.fact_checker = FactChecker(table_df)
        self.z3_auditor = Z3Auditor(table_df)

def run(self, trace: CoTTrace) -> Tuple[bool, Optional[dict]]:
        logger.info(f"Starting verification for Q: {trace.question}")
        verified_facts = []
        for step in trace.steps:
            if step.step_type == "fact":
                res = self.fact_checker.verify(step, context=verified_facts)
            else:
                res = self.z3_auditor.verify(step, context=verified_facts)

            if not res.is_valid:
                return False, {
                    "step_index": step.step_id,
                    "module": res.component,  # FactChecker / Z3Auditor
                    "reason": res.reason,
                    "counter_example": getattr(res, "counter_example", None) 
                }
            verified_facts.append(step)

        declared_answer = str(trace.final_answer).lower().strip()
        last_step_content = str(trace.steps[-1].content).lower().strip()

        if declared_answer and declared_answer not in last_step_content:
             return False, {
                "step_index": -1,
                "module": "ConsistencyMonitor",
                "reason": f"Execution Inconsistency: Derived '{last_step_content}' != Answer '{declared_answer}'"
             }

        return True, None