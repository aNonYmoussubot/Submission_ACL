import json
import os
from collections import defaultdict
import pandas as pd

def evaluate_verifier_metrics(result_file):

    if not os.path.exists(result_file):
        print(f"Error: File {result_file} not found.")
        return

    print(f"Loading results from: {result_file}")
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # stats[std_type] = {'total': 0, 'accept': 0, 'reject': 0}
    stats = defaultdict(lambda: {"total": 0, "accept": 0, "reject": 0})
    
    subtype_stats = defaultdict(lambda: {"total": 0, "accept": 0, "reject": 0})

    type_mapping = {
        "type1_golden": "Type 1 (Faithful)",
        "type2_spurious": "Type 2 (Spurious)", 
        "type4_calc_error": "Type 4 (Inconsistent)",
        "type3_fully_wrong": "Type 3 (Wrong)" 
    }

    total_accepted_count = 0
    type1_accepted_count = 0

    for item in results:
        raw_type = item.get("target_type", "unknown")
        sub_type = item.get("specific_subtype", "unknown")
        
        decision = item.get("verifier_decision", "UNKNOWN")
        
        std_type = type_mapping.get(raw_type, "Unknown Type")
        
        stats[std_type]["total"] += 1

        subtype_stats[sub_type]["total"] += 1
        
        if decision == "ACCEPT":
            stats[std_type]["accept"] += 1
            subtype_stats[sub_type]["accept"] += 1
            
            total_accepted_count += 1
            if std_type == "Type 1 (Faithful)":
                type1_accepted_count += 1
                
        elif decision == "REJECT":
            stats[std_type]["reject"] += 1
            subtype_stats[sub_type]["reject"] += 1
        


    
    # --- Metric 1: VCAR (Verified Correct Answer Rate) ---
    # Formula: Accept(Type 1) / Total(Type 1)
    t1_stats = stats.get("Type 1 (Faithful)", {"total": 0, "accept": 0})
    vcar = 0.0
    if t1_stats["total"] > 0:
        vcar = (t1_stats["accept"] / t1_stats["total"]) * 100

    # --- Metric 2: DIR (Diagnostic Interception Rate) ---
    # Formula: Reject(Type k) / Total(Type k)
    
    # DIR_spur (Type 2 - Aggregated)
    t2_stats = stats.get("Type 2 (Spurious)", {"total": 0, "reject": 0})
    dir_spur = 0.0
    if t2_stats["total"] > 0:
        dir_spur = (t2_stats["reject"] / t2_stats["total"]) * 100
        
    # DIR_inc (Type 4)
    t4_stats = stats.get("Type 4 (Inconsistent)", {"total": 0, "reject": 0})
    dir_inc = 0.0
    if t4_stats["total"] > 0:
        dir_inc = (t4_stats["reject"] / t4_stats["total"]) * 100

    # --- Metric 3: FP (Faithfulness Precision) ---
    # Formula: Accept(Type 1) / Total_Accepted_All_Types
    fp = 0.0
    if total_accepted_count > 0:
        fp = (type1_accepted_count / total_accepted_count) * 100

    print("\n" + "="*80)
    print(f"{'METRIC':<25} | {'VALUE':<10} | {'DEFINITION & GOAL'}")
    print("-" * 80)
    
    print(f"{'VCAR (Faithfulness)':<25} | {vcar:5.1f}%     | Retention of Type 1. (Ideal: High)")
    print(f"{'DIR_spur (Safety)':<25} | {dir_spur:5.1f}%     | Interception of Type 2. (Ideal: High)")
    print(f"{'DIR_inc (Consistency)':<25} | {dir_inc:5.1f}%     | Interception of Type 4. (Ideal: High)")
    print(f"{'FP (Trustworthiness)':<25} | {fp:5.1f}%     | Precision of Acceptance. (Ideal: 100%)")
    
    print("-" * 80)
    print(">>> Detailed Breakdown by Subtype (Debug Info):")
    print(f"{'Subtype':<30} | {'Total':<6} | {'Accepted':<8} | {'Rejected':<8} | {'DIR (%)'}")
    print("-" * 80)
    
    sorted_subtypes = sorted(subtype_stats.items(), key=lambda x: x[0])
    for sub, data in sorted_subtypes:
        total = data["total"]
        if total == 0: continue
        d_rate = (data["reject"] / total) * 100
        print(f"{sub:<30} | {total:<6} | {data['accept']:<8} | {data['reject']:<8} | {d_rate:5.1f}%")

    print("="*80 + "\n")

    return {
        "VCAR": vcar,
        "DIR_spur": dir_spur,
        "DIR_inc": dir_inc,
        "FP": fp
    }

if __name__ == "__main__":

    RESULT_FILE = "./output/deepseek/fin_cot_verifier_results.json"
    evaluate_verifier_metrics(RESULT_FILE)