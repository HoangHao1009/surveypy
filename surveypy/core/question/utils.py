import pandas as pd

def _melt_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
    """Handle the melting of the DataFrame as required by df_config."""
    df = df.reset_index().melt(id_vars='resp_id')
    df = df[df['value'] != 0]
    df = df.add_prefix(f'{self.code}_')
    df.rename(columns={f'{self.code}_resp_id': 'resp_id'}, inplace=True)
    return df

def _get_duplicates(lst):
    duplicates = []
    seen = set()
    for item in lst:
        if item in seen and item not in duplicates:
            duplicates.append(item)
        seen.add(item)
    return duplicates
