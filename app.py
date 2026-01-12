import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np

# ==============================================================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Olist Logistics AI",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. FUNÇÕES DE CARGA (CACHE)
# ==============================================================================


@st.cache_resource
def load_model():
    return joblib.load('models/modelo_previsao_atraso_olist.pkl')


@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/data_dashboard_processed.csv')
        return df
    except FileNotFoundError:
        return None


model = load_model()
df = load_data()

# ==============================================================================
# 3. SIDEBAR (FILTROS GERAIS)
# ==============================================================================

st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/8/83/Olist_logo.svg/1200px-Olist_logo.svg.png", width=150)
st.sidebar.header("Filtros Globais")

if df is not None:
    estados = sorted(df['uf_cliente'].unique())
    filtro_uf = st.sidebar.multiselect(
        "Filtrar Estado Cliente:", estados, default=estados[:2])

    categorias = sorted(df['categoria_produto'].unique())
    filtro_cat = st.sidebar.multiselect("Filtrar Categoria:", categorias)

    df_filtered = df.copy()
    if filtro_uf:
        df_filtered = df_filtered[df_filtered['uf_cliente'].isin(filtro_uf)]
    if filtro_cat:
        df_filtered = df_filtered[df_filtered['categoria_produto'].isin(
            filtro_cat)]

# ==============================================================================
# 4. INTERFACE PRINCIPAL (ABAS)
# ==============================================================================

tab1, tab2 = st.tabs(["📈 Visão Gerencial (Dashboard)",
                     "🤖 Simulador de Atraso (IA)"])

# --- TAB 1: DASHBOARD GERENCIAL ---
with tab1:
    if df is None:
        st.error(
            "Arquivo 'dataset_dashboard.csv' não encontrado. Exporte o CSV do notebook para a pasta data/.")
    else:
        st.markdown("### 📊 KPIs de Logística")

        # Métricas (Cards)
        col1, col2, col3, col4 = st.columns(4)

        total_pedidos = len(df_filtered)
        taxa_atraso = (df_filtered[df_filtered['dias_atraso'] >
                       0].shape[0] / total_pedidos) * 100 if total_pedidos > 0 else 0
        media_atraso = df_filtered[df_filtered['dias_atraso']
                                   > 0]['dias_atraso'].mean()

        col1.metric("Total de Pedidos", f"{total_pedidos:,}")
        col2.metric("Taxa de Atraso",
                    f"{taxa_atraso:.1f}%", delta_color="inverse")
        col3.metric("Média de Atraso (dias)",
                    f"{media_atraso:.2f}", delta_color="inverse")
        col4.metric("Performance SLA", f"{100-taxa_atraso:.1f}%")

        st.markdown("---")

        # Gráficos
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.markdown("#### 🌍 Atraso Médio por Estado (Destino)")
            df_uf = df_filtered.groupby('uf_cliente')['dias_atraso'].mean(
            ).reset_index().sort_values('dias_atraso', ascending=False)
            fig_map = px.bar(df_uf.head(10), x='dias_atraso', y='uf_cliente', orientation='h',
                             title="Top 10 Estados com Maior Atraso Médio", color='dias_atraso', color_continuous_scale='Reds')
            st.plotly_chart(fig_map, use_container_width=True)

        with col_g2:
            st.markdown("#### 📦 Atraso por Categoria")
            df_cat = df_filtered.groupby('categoria_produto')['dias_atraso'].mean(
            ).reset_index().sort_values('dias_atraso', ascending=False)
            fig_cat = px.bar(df_cat.head(10), x='categoria_produto', y='dias_atraso',
                             title="Top 10 Categorias Problemáticas", color='dias_atraso', color_continuous_scale='OrRd')
            st.plotly_chart(fig_cat, use_container_width=True)

# --- TAB 2: SIMULADOR IA ---
with tab2:
    st.markdown("### 🔮 Simulador de Risco de Atraso")
    st.markdown("Preencha os dados do pedido para prever se haverá atraso.")

    with st.form("form_simulador"):
        col_s1, col_s2, col_s3 = st.columns(3)

        with col_s1:
            uf_vendedor = st.selectbox(
                "UF Vendedor", ['SP', 'PR', 'MG', 'RJ', 'SC', 'RS', 'Outros'])
            uf_cliente = st.selectbox(
                "UF Cliente", ['SP', 'RJ', 'MG', 'RS', 'PR', 'SC', 'BA', 'Outros'])
            prazo_prometido = st.number_input(
                "Prazo Prometido (dias)", min_value=1, value=5)

        with col_s2:
            categoria = st.selectbox("Categoria", ['cama_mesa_banho', 'beleza_saude', 'esporte_lazer',
                                     'moveis_decoracao', 'informatica_acessorios', 'utilidades_domesticas', 'outros'])
            peso_g = st.number_input("Peso Real (g)", min_value=10, value=500)
            tempo_aprovacao = st.number_input(
                "Tempo Aprovação (dias)", min_value=0, value=0)

        with col_s3:
            # Inputs auxiliares para calcular volume e peso cubado
            st.write("Dimensões da Embalagem (cm):")
            comp = st.number_input("Comprimento", 20)
            larg = st.number_input("Largura", 20)
            alt = st.number_input("Altura", 10)
            flag_pickup = st.selectbox("Ponto de Retirada (Pickup)?", [
                                       0, 1], format_func=lambda x: "Sim" if x == 1 else "Não")

        # Botão de Calcular
        submit = st.form_submit_button("🚨 Calcular Previsão")

    if submit:
        # 1. Calculando as features derivadas (Engenharia Reversa)
        vol_cm3 = comp * larg * alt
        peso_cubado_kg = vol_cm3 / 6000  # Fator de cubagem padrão

        # 2. Criando o DataFrame com os dados do input
        input_data = pd.DataFrame({
            'peso_cubado_kg': [peso_cubado_kg],
            'vol_cm3': [vol_cm3],
            'peso_produto_g': [peso_g],
            'tempo_aprovacao_dias': [tempo_aprovacao],
            'prazo_prometido_dias': [prazo_prometido],
            'uf_vendedor': [uf_vendedor],
            'uf_cliente': [uf_cliente],
            'flag_pickup': [flag_pickup],
            'categoria_produto': [categoria]
        })

        # 3. Previsão
        try:
            predicao_dias = model.predict(input_data)[0]

            # Mostrando o resultado
            st.divider()
            col_res1, col_res2 = st.columns([1, 2])

            with col_res1:
                if predicao_dias > 0:
                    st.error(f"⚠️ Atraso Previsto!")
                    st.metric("Dias de Atraso", f"{predicao_dias:.2f}")
                else:
                    st.success(f"✅ Entrega no Prazo!")
                    st.metric("Margem de Segurança",
                              f"{abs(predicao_dias):.2f} dias")

            with col_res2:
                st.info(
                    f"💡 **Interpretação:** O modelo estima que, para este cenário, a entrega ocorrerá em **{prazo_prometido + predicao_dias:.1f} dias** totais.")
                if predicao_dias > 0:
                    st.warning(
                        "Sugestão: Aumente o prazo prometido ou verifique o estoque para despacho imediato.")

        except Exception as e:
            st.error(f"Erro na previsão: {e}")
