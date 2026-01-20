import z3
from src.verifiers.base import BaseVerifier
from src.schema import ReasoningStep, VerificationResult
from src.llm_engine import LLMEngine
from utils.logger import setup_logger

logger = setup_logger("Z3Auditor")

class Z3Auditor(BaseVerifier):
    def __init__(self, table):
        super().__init__(table)
        self.llm = LLMEngine()



    def verify(self, step: ReasoningStep, context: list) -> VerificationResult:
        if step.step_type != "inference":
            return VerificationResult(True, "Z3Auditor", "Skipping.")

        verified_facts = [s.content for s in context if s.step_type == "fact"]
        premise_text = "\n".join(verified_facts) if verified_facts else "No factual context"
        conclusion_text = step.content
        
        table_str = self.table.to_csv(sep="|", index=False)

        logger.info(f"Auditing with FULL Table Context ({len(self.table)} rows)...")

        z3_code = self.llm.autoformalize_to_z3(premise_text, conclusion_text, table_str)
        logger.debug(f"Generated Z3 Code:\n{z3_code}")

        try:
            exec_globals = {
                "Distinct": z3.Distinct, 
                
                "Const": z3.Const,        
                "Function": z3.Function,  
                "z3": z3,
                "Solver": z3.Solver,
                "Int": z3.Int,
                "Ints": z3.Ints,
                "String": z3.String,
                "Real": z3.Real,   
                "Bool": z3.Bool,   
                "Not": z3.Not,
                "StringVal": z3.StringVal,
                "And": z3.And,    
                "Or": z3.Or,      
                "Implies": z3.Implies, 
                "If": z3.If,       
                "sat": z3.sat,
                "unsat": z3.unsat
            }
            exec_locals = {}
            
            exec(z3_code, exec_globals, exec_locals)
            
            if "solve_logic" not in exec_locals:
                raise ValueError("LLM did not generate 'solve_logic' function.")
            
            is_valid, model = exec_locals["solve_logic"]()
            
            if is_valid:
                return VerificationResult(True, "Z3Auditor", "Logic is mathematically sound.")
            else:
                return VerificationResult(
                    False, 
                    "Z3Auditor", 
                    "Logic Error: Counter-example found (Spurious Correlation detected).",
                    counter_example=str(model)
                )

        except Exception as e:
            logger.error(f"Z3 Execution Failed: {e}")
            return VerificationResult(False, "Z3Auditor", f"Symbolic Execution Error: {e}")
    def verify11(self, step: ReasoningStep, context: list) -> VerificationResult:

        if step.step_type != "inference":
            return VerificationResult(True, "Z3Auditor", "Skipping non-inference step.")

        all_verified_facts = [s.content for s in context if s.step_type == "fact"]
        premise_text = "\n".join(all_verified_facts) if all_verified_facts else "No factual context"
        #premise_text = context[-1].content if context else "No context"
        conclusion_text = step.content
        
        logger.info(f"Auditing Logic: {premise_text} => {conclusion_text}")

        z3_code = self.llm.autoformalize_to_z3(premise_text, conclusion_text)
        logger.debug(f"Generated Z3 Code:\n{z3_code}")

        try:
            exec_globals = {
                "z3": z3,
                "Solver": z3.Solver,
                "Int": z3.Int,
                "Ints": z3.Ints,
                "Real": z3.Real,   
                "Bool": z3.Bool,   
                "Not": z3.Not,
                "And": z3.And,    
                "Or": z3.Or,      
                "Implies": z3.Implies,
                "If": z3.If,       
                "sat": z3.sat,
                "unsat": z3.unsat
            }
            exec_locals = {}
            
            exec(z3_code, exec_globals, exec_locals)
            
            if "solve_logic" not in exec_locals:
                raise ValueError("LLM did not generate 'solve_logic' function.")
            
            is_valid, model = exec_locals["solve_logic"]()
            
            if is_valid:
                return VerificationResult(True, "Z3Auditor", "Logic is mathematically sound.")
            else:
                return VerificationResult(
                    False, 
                    "Z3Auditor", 
                    "Logic Error: Counter-example found (Spurious Correlation detected).",
                    counter_example=str(model)
                )

        except Exception as e:
            logger.error(f"Z3 Execution Failed: {e}")
            return VerificationResult(False, "Z3Auditor", f"Symbolic Execution Error: {e}")