import pandas as pd
from src.verifiers.base import BaseVerifier
from src.schema import ReasoningStep, VerificationResult
from src.llm_engine import LLMEngine
from utils.logger import setup_logger

logger = setup_logger("FactChecker")

class FactChecker(BaseVerifier):
    def __init__(self, table: pd.DataFrame):
        super().__init__(table)
        self.llm = LLMEngine()

    def verify(self, step: ReasoningStep, context: list) -> VerificationResult:
        content = step.content
        logger.info(f"Fact Checking: \"{content}\"")

        columns = self.table.columns.tolist()
        sample_row = self.table.head(3).to_dict(orient='records')
        
        code = self.llm.generate_pandas_check(content, columns, str(sample_row))
        # logger.debug(f"Generated Pandas Code:\n{code}")

        try:
            exec_globals = {'pd': pd}
            exec_locals = {}
            
            exec(code, exec_globals, exec_locals)
            
            if 'verify_fact' not in exec_locals:
                return VerificationResult(False, "FactChecker", "LLM failed to generate 'verify_fact' function.")
            
            verify_func = exec_locals['verify_fact']
            is_fact_true = verify_func(self.table)
            
            if is_fact_true:
                return VerificationResult(True, "FactChecker", "Data Grounding Successful.")
            else:
                return VerificationResult(False, "FactChecker", "Data Mismatch: Table data contradicts the claim.")
                
        except Exception as e:
            logger.warning(f"Pandas Execution Error: {e}")
            return VerificationResult(False, "FactChecker", f"Grounding Error (Execution Failed): {str(e)}")