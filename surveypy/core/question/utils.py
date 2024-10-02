import pandas as pd

def _melt_dataframe(code, df: pd.DataFrame) -> pd.DataFrame:
    """Handle the melting of the DataFrame as required by df_config."""
    try:
        df = df.reset_index().melt(id_vars='resp_id')
    except:
        print("MELT ERROR: ", code)
        print(df.reset_index())
        print(df.columns)
    df = df[df['value'] != 0]
    df = df.add_prefix(f'{code}_')
    df.rename(columns={f'{code}_resp_id': 'resp_id'}, inplace=True)
    return df

def _get_duplicates(lst):
    duplicates = []
    seen = set()
    for item in lst:
        if item in seen and item not in duplicates:
            duplicates.append(item)
        seen.add(item)
    return duplicates
