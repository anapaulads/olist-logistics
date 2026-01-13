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


# CONSTANTES (Dicionários de Mapeamento)

map_cat_correcao = {
    'eletrodomesticos_2': 'eletrodomesticos',
    'casa_conforto_2': 'casa_conforto',
    'pc_gamer': 'pcs',
    'ferramentas_jardim': 'construcao_ferramentas_jardim'
}

map_status_simplificado = {
    "delivered": "Aprovado",
    "shipped": "Aprovado",
    "approved": "Em Processamento",
    "processing": "Em Processamento",
    "invoiced": "Em Processamento",
    "canceled": "Cancelado",
    "unavailable": "Cancelado"
}

# FUNÇÕES DE TRATAMENTO


def tratar_categorias(df, col_cat='categoria_produto'):
    if col_cat not in df.columns:
        return df
    df[col_cat] = df[col_cat].replace(map_cat_correcao)
    df['categoria_label'] = df[col_cat].astype(
        str).str.replace('_', ' ').str.title()
    return df


def tratar_status(df, col_status='order_status'):
    col_alvo = col_status
    if col_status not in df.columns and 'status_pedido' in df.columns:
        col_alvo = 'status_pedido'

    if col_alvo in df.columns:
        df["status_simplificado"] = df[col_alvo].map(
            map_status_simplificado)
    else:
        df["status_simplificado"] = "N/A"
    return df


def enriquecer_dados_dash(df):
    """
    Pipeline completo de tratamento para o Dashboard.
    """
    df = tratar_categorias(df)
    df = tratar_status(df)

    # KPIs Calculados
    if 'preco_produto' in df.columns and 'valor_frete' in df.columns:
        df["faturamento_pedido"] = df["preco_produto"] + df["valor_frete"]

    if 'dias_atraso' in df.columns:
        df["flag_atraso"] = df["dias_atraso"] > 0

    # --- AQUI ESTAVA FALTANDO O CÁLCULO DO TEMPO DE PROCESSAMENTO ---
    # Necessário para o gráfico de barras
    if 'data_postagem' in df.columns and 'data_aprovacao' in df.columns:
        # Garante que são datas
        df['data_postagem'] = pd.to_datetime(df['data_postagem'])
        df['data_aprovacao'] = pd.to_datetime(df['data_aprovacao'])

        # Calcula a diferença em dias
        df["tempo_processamento"] = (
            df["data_postagem"] - df["data_aprovacao"]).dt.total_seconds() / 86400
    else:
        df["tempo_processamento"] = 0

    return df
