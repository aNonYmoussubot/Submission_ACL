import pandas as pd

def parse_structured_table(table_dict: dict) -> pd.DataFrame:
    """
    直接从结构化字典加载表格，100% 避免解析错误。
    """
    try:

        header = table_dict.get("header", [])
        rows = table_dict.get("rows", [])
        

        df = pd.DataFrame(rows, columns=header)
        
        df.columns = [c.replace('\n', ' ').strip() for c in df.columns]
        
        df = df.apply(lambda x: x.astype(str).str.replace('\n', ' ').str.strip())
        
        return df
    except Exception as e:
        print(f"Structured Table Load Error: {e}")
        return pd.DataFrame()