import json
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional


from src.pipeline import TrustTablePipeline
from src.schema import CoTTrace, ReasoningStep
from src.llm_engine import LLMEngine
from src.refiner import BlindIterativeRefiner
from utils.logger import setup_logger
from utils.table_utils import parse_structured_table


@dataclass
class EvalStats:

    total_samples: int = 0
    

    count_type1: int = 0  # Faithful (Z+ A+)
    count_type2: int = 0  # Spurious (Z- A+)
    count_type3: int = 0  # Wrong (Z- A-)
    count_type4: int = 0  # Inconsistent (Z+ A-) 
    

    accepted_type1: int = 0 # True Positive
    accepted_type2: int = 0 # False Positive (Spurious leak)
    accepted_type3: int = 0 # False Positive (Wrong leak)
    accepted_type4: int = 0 # False Positive (Inconsistent leak)
    
    rejected_total: int = 0 #  (Initial Rejections)
    

    repaired_to_type1: int = 0 #  (Refined -> Accepted & Correct)

    def print_latex_report(self):
        """生成符合 LaTeX 定义的指标报告"""
        # 1. VCAR (Verified Correct Answer Rate)
        # Definition: P(Accept AND Type1) over Total Dataset
        vcar = self.accepted_type1 / self.total_samples if self.total_samples > 0 else 0
        
        # 2. DIR (Diagnostic Interception Rate)
        # DIR_spur = Reject Rate on Type 2
        dir_spur = (self.count_type2 - self.accepted_type2) / self.count_type2 if self.count_type2 > 0 else 0
        # DIR_inc = Reject Rate on Type 4
        dir_inc = (self.count_type4 - self.accepted_type4) / self.count_type4 if self.count_type4 > 0 else 0
        
        # 3. FP (Faithfulness Precision)
        # Definition: Accepted Type 1 / Total Accepted
        total_accepted = (self.accepted_type1 + self.accepted_type2 + 
                          self.accepted_type3 + self.accepted_type4)
        fp = self.accepted_type1 / total_accepted if total_accepted > 0 else 0
        
        # 4. CSR (Correction Success Rate)
        # Definition: Refined to Type 1 / Initial Rejections
        csr = self.repaired_to_type1 / self.rejected_total if self.rejected_total > 0 else 0

        print("\n" + "="*60)
        print("📊 RIGOROUS EVALUATION REPORT (Based on App. Metrics)")
        print("="*60)
        print(f"Total Samples (N): {self.total_samples}")
        print(f"  - Type 1 (Faithful)     : {self.count_type1}")
        print(f"  - Type 2 (Spurious)     : {self.count_type2}")
        print(f"  - Type 3 (Wrong)        : {self.count_type3}")
        print(f"  - Type 4 (Inconsistent) : {self.count_type4}")
        print("-" * 60)
        
        print(f"1. [VCAR] Verified Correct Answer Rate : {vcar:.2%} (Target: High)")
        print(f"   (Strict Recall of Faithfulness)")
        
        print(f"2. [DIR] Diagnostic Interception Rate  :")
        print(f"   - DIR_spur (Defend vs Type 2)       : {dir_spur:.2%} (Target: High)")
        print(f"   - DIR_inc  (Defend vs Type 4)       : {dir_inc:.2%}  (Target: High)")
        
        print(f"3. [FP] Faithfulness Precision         : {fp:.2%} (Target: High)")
        print(f"   (Trustworthiness of Acceptance)")
        
        print(f"4. [CSR] Correction Success Rate       : {csr:.2%} (Target: >0)")
        print(f"   (Self-Repair Efficiency: {self.repaired_to_type1}/{self.rejected_total})")
        print("="*60)


def is_answer_correct(pred: str, gold: str) -> bool:
    """简易的答案比对逻辑，实际项目可用更复杂的 Normalization"""
    if not pred: return False
    p = str(pred).lower().strip().replace(".0", "")
    g = str(gold).lower().strip().replace(".0", "")
    return p == g or p in g or g in p


def run_experiment():
    logger = setup_logger("Evaluation")
    

    data_path = 'data/wtq_qa_merged_all.json'
    dataset = []
    
    if os.path.exists(data_path):
        logger.info(f"Loading dataset from {data_path}...")
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                full_data = json.load(f)

                dataset = full_data 
        except Exception as e:
            logger.error(f"Failed to load file: {e}")
            return
    else:
        logger.error(f"Dataset not found at {data_path}")
        return

    llm_engine = LLMEngine()
    stats = EvalStats()

    logger.info(">>> STARTING EVALUATION LOOP <<<")
    
    for case_idx, data in enumerate(dataset):
        case_id = data.get('id', f'case_{case_idx}')
        gold_answer = str(data.get('gold_answer', ''))
        

        try:
            df = parse_structured_table(data['table_content'])
        except Exception as e:
            logger.error(f"Table parsing error for {case_id}: {e}")
            continue

        pipeline = TrustTablePipeline(df)
        refiner = BlindIterativeRefiner(df, llm_engine, refinement_enabled=True)
        
        samples = data.get('generated_samples', {})
        

        for sample_key, sample_data in samples.items():

            cot_text = sample_data.get('chain_of_thought') or \
                       sample_data.get('flawed_chain_of_thought') or \
                       sample_data.get('incorrect_chain_of_thought') or \
                       sample_data.get('correct_logic_wrong_math_cot') 
            
            if not cot_text:

                continue

            gt_type = 0
            if "type1" in sample_key:
                gt_type = 1
            elif "type2" in sample_key:
                gt_type = 2
            elif "type3" in sample_key:
                gt_type = 3
            elif "type4" in sample_key:
                gt_type = 4
            else:
                logger.warning(f"Unknown sample type: {sample_key}")
                continue
            
            # 更新分母
            stats.total_samples += 1
            if gt_type == 1: stats.count_type1 += 1
            elif gt_type == 2: stats.count_type2 += 1
            elif gt_type == 3: stats.count_type3 += 1
            elif gt_type == 4: stats.count_type4 += 1

            logger.info(f"--- Sample: {sample_key} (GT: Type {gt_type}) ---")

            # ==========================================================
            # PHASE 1: INITIAL VERIFICATION
            # ==========================================================
            steps = llm_engine.decompose_cot(cot_text)
            trace = CoTTrace(
                question=data['original_question'],
                steps=[ReasoningStep(i+1, s['content'], s['type']) for i, s in enumerate(steps)],
                final_answer=gold_answer 
            )

            is_valid, error_report = pipeline.run(trace)
            

            if is_valid:
                logger.info(f"Verdict: ACCEPTED")
                if gt_type == 1: stats.accepted_type1 += 1
                elif gt_type == 2: stats.accepted_type2 += 1
                elif gt_type == 3: stats.accepted_type3 += 1
                elif gt_type == 4: stats.accepted_type4 += 1
            else:
                logger.info(f"Verdict: REJECTED | Reason: {error_report.get('reason')}")
                stats.rejected_total += 1 # 记录 CSR 分母

            # ==========================================================
            # PHASE 2: REFINEMENT (Only if Rejected)
            # ==========================================================
            if not is_valid:
                logger.info("🔧 Triggering Refinement...")
                

                repaired_cot = refiner._refine_cot(data['original_question'], cot_text, error_report)
                

                new_steps = llm_engine.decompose_cot(repaired_cot)

                refined_answer = refiner._extract_answer(repaired_cot)
                
                new_trace = CoTTrace(
                    question=data['original_question'],
                    steps=[ReasoningStep(i+1, s['content'], s['type']) for i, s in enumerate(new_steps)],
                    final_answer=refined_answer
                )
                
                repaired_valid, _ = pipeline.run(new_trace)
                

                if repaired_valid:
                    answer_match = is_answer_correct(refined_answer, gold_answer)
                    if answer_match:
                        stats.repaired_to_type1 += 1
                        logger.info(f"✨ CSR Hit: Refined to Valid Logic & Correct Answer '{refined_answer}'")
                    else:
                        logger.info(f"⚠️ Refined Logic Valid, but Answer Wrong ('{refined_answer}' != '{gold_answer}'). Not Type 1.")
                else:
                    logger.info("❌ Refinement Failed: Still Invalid.")


    stats.print_latex_report()

if __name__ == "__main__":
    run_experiment()