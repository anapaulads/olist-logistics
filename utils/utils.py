import pandas as pd


def formatar_datetime(df, prefixo="data"):
    """
    Converte para datetime todas as colunas de um DataFrame
    que começam com o prefixo especificado (default = 'data').

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada
    prefixo : str, optional
        Prefixo usado para identificar colunas (default 'data')

    Returns
    -------
    pd.DataFrame
        DataFrame com colunas convertidas para datetime
    """
    cols_data = [col for col in df.columns if col.startswith(prefixo)]
    for col in cols_data:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df
