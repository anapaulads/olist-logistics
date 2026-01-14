from utils.utils import enriquecer_dados_dash, map_cat_correcao
import streamlit as st
import pandas as pd
import plotly.express as px
import json
import joblib
import numpy as np
import os
import sys

sys.path.append(os.path.abspath('.'))

# ==============================================================================
# 1. CONFIGURAÇÃO DA PÁGINA E ESTILO
# ==============================================================================
st.set_page_config(
    page_title="Olist Logistica Control Tower",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cores do Tema
PRIMARY = "#0B16B3"
SECONDARY = "#583a89"
WARNING = "#ff4b4b"

# CSS Personalizado para métricas e títulos
st.markdown(f"""
<style>
    /* Estilo dos Cards de Métricas */
    div[data-testid="metric-container"] {{
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid {PRIMARY};
        box-shadow: 2px 2px 5px rgba(0,0,0,0.08);
    }}
    /* Títulos Coloridos */
    h1, h2, h3 {{ color: {PRIMARY}; }}
    
    /* Ajuste para textos em fundo escuro nos gráficos Plotly */
    g.pointtext {{ fill: white !important; }}
    
    /* Botões */
    div.stButton > button {{
        background-color: {PRIMARY};
        color: white;
        border-radius: 8px;
        width: 100%;
    }}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CARREGAMENTO DE DADOS E MODELO (CACHED)
# ==============================================================================


@st.cache_data
def load_data():
    """Carrega e trata os dados para o Dashboard."""
    caminho_dados = "data/data_dashboard_processed.csv"

    if not os.path.exists(caminho_dados):
        st.error(f"Arquivo não encontrado: {caminho_dados}")
        return pd.DataFrame()

    df = pd.read_csv(
        caminho_dados,
        parse_dates=["data_aprovacao", "data_postagem"]
    )

    # Aplica o tratamento de categorias
    if 'status_simplificado' not in df.columns:
        df = enriquecer_dados_dash(df)

    return df


@st.cache_resource
def load_model():
    """Carrega o modelo treinado."""
    caminho_modelo = 'models/modelo_previsao_atraso_olist.pkl'
    if not os.path.exists(caminho_modelo):
        st.error("Modelo não encontrado. Verifique a pasta 'models'.")
        return None
    return joblib.load(caminho_modelo)


df = load_data()
model = load_model()

# ==============================================================================
# 3. PREPARAÇÃO DE MAPEAMENTOS AUXILIARES
# ==============================================================================
if not df.empty and 'categoria_label' in df.columns:
    mapa_reverso_categorias = dict(
        zip(df['categoria_label'], df['categoria_produto']))
else:
    # Fallback caso não tenha a coluna label
    mapa_reverso_categorias = {
        cat: cat for cat in df['categoria_produto'].unique()}


def abreviar_valor(valor):
    """Formata números grandes (1M, 1K)."""
    if valor >= 1_000_000:
        return f"{valor/1_000_000:.1f}M"
    if valor >= 1_000:
        return f"{valor/1_000:.1f}K"
    return f"{valor:.0f}"


# ==============================================================================
# 4. SIDEBAR (FILTROS)
# ==============================================================================
st.sidebar.image(
    "https://d3hw41hpah8tvx.cloudfront.net/images/logo_ecossistema_66f532e37b.svg",
    width=180
)
st.sidebar.divider()
st.sidebar.header("Filtros Globais")

# Filtro 1: Status
if not df.empty:
    status_options = sorted(df["status_simplificado"].astype(str).unique())
    status_sel = st.sidebar.multiselect("Status do Pedido", status_options)

    # Filtro 2: Estado
    estados_options = sorted(df["uf_cliente"].unique())
    estados_sel = st.sidebar.multiselect("Estado do Cliente", estados_options)

    # Filtro 3: Categoria
    cat_options = sorted(df["categoria_label"].unique())
    cat_sel = st.sidebar.multiselect("Categoria de Produto", cat_options)

    # Aplicação dos Filtros
    df_f = df.copy()

    if status_sel:
        df_f = df_f[df_f["status_simplificado"].isin(status_sel)]
    if estados_sel:
        df_f = df_f[df_f["uf_cliente"].isin(estados_sel)]
    if cat_sel:
        df_f = df_f[df_f["categoria_label"].isin(cat_sel)]
else:
    st.sidebar.warning("Sem dados carregados.")
    df_f = pd.DataFrame()


def abreviar(valor):
    if valor >= 1_000_000:
        return f"{valor/1_000_000:.1f}M"
    if valor >= 1_000:
        return f"{valor/1_000:.1f}K"
    return f"{valor:.0f}"


st.sidebar.info(f"Visualizando {len(df_f)} pedidos filtrados.")

# ==============================================================================
# 5. ESTRUTURA DE ABAS
# ==============================================================================
tab_analise, tab_previsao = st.tabs(
    ["📊 Visão Geral da Operação", "🔮 Simulador de Atraso (IA)"])

# ------------------------------------------------------------------------------
# ABA 1: VISÃO GERAL (DASHBOARD)
# ------------------------------------------------------------------------------
with tab_analise:
    st.markdown("## 📊 Visão Geral da Operação")

    col1, col2, col3, col4, col5 = st.columns(5)

    total_pedidos = len(df_f)
    faturamento = df_f["faturamento_pedido"].sum()
    taxa_atraso = df_f["flag_atraso"].mean() * 100 if total_pedidos > 0 else 0
    atraso_medio = df_f.loc[df_f["flag_atraso"], "dias_atraso"].mean()
    taxa_cancel = (df_f["status_simplificado"] ==
                   "Cancelado").mean() * 100 if total_pedidos > 0 else 0

    col1.metric("📦 Pedidos", abreviar(total_pedidos))
    col2.metric("💰 Faturamento", f"R$ {abreviar(faturamento)}")
    col3.metric("🚨 Taxa de Atraso", f"{taxa_atraso:.1f}%")
    col4.metric("⏱️ Atraso Médio", f"{atraso_medio:.1f} dias" if not pd.isna(
        atraso_medio) else "0 dias")
    col5.metric("🚫 Cancelamento", f"{taxa_cancel:.1f}%")

    st.divider()

    colA, colB = st.columns([1, 1.5])

    with colA:
        st.markdown("##### Status dos Pedidos")
        df_status = df_f["status_simplificado"].value_counts().reset_index()
        df_status.columns = ["Status", "Pedidos"]

        fig_status = px.pie(
            df_status,
            names="Status",
            values="Pedidos",
            hole=0.6,
            color="Status",
            color_discrete_map={
                "Aprovado": PRIMARY,
                "Em Processamento": SECONDARY,
                "Cancelado": WARNING,
            }
        )
        fig_status.update_layout(
            legend=dict(orientation="h", yanchor="top",
                        y=-0.25, xanchor="center", x=0.5),
            margin=dict(t=60, b=60, l=10, r=10),
            height=300,
            font=dict(color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig_status.update_traces(textfont_color='white')
        st.plotly_chart(fig_status, use_container_width=True)

    with colB:
        st.markdown("##### Intensidade de Atrasos (Brasil)")
        try:
            with open("data/brazil_states.geojson", "r", encoding="utf-8") as f:
                brazil_geojson = json.load(f)

            df_geo = df_f[df_f["dias_atraso"] > 0].groupby("uf_cliente").agg(
                pedidos=("pedido_id", "count"),
                atraso_medio=("dias_atraso", "mean")
            ).reset_index()

            fig_map = px.choropleth(
                df_geo,
                geojson=brazil_geojson,
                locations="uf_cliente",
                featureidkey="properties.sigla",
                color="atraso_medio",
                color_continuous_scale="Reds",
                title="Média de Dias de Atraso"
            )
            fig_map.update_geos(fitbounds="locations", visible=False)
            fig_map.update_layout(
                margin=dict(l=0, r=0, t=30, b=0),
                height=300,
                font=dict(color='white'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_map, use_container_width=True)

        except Exception:
            st.info("Mapa indisponível (GeoJSON não encontrado).")

    st.divider()

    # Gráficos Linha 2
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Top 10 Estados (Faturamento)")
        df_fat_estado = df_f.groupby("uf_cliente")["faturamento_pedido"].sum(
        ).reset_index().sort_values("faturamento_pedido", ascending=False).head(10)

        fig_fat = px.bar(df_fat_estado, x="uf_cliente",
                         y="faturamento_pedido", text_auto='.2s')

        fig_fat.update_traces(marker_color=PRIMARY, textfont_color='white')
        fig_fat.update_layout(
            xaxis_title=None, yaxis_title=None,
            font=dict(color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_fat, use_container_width=True)

    with col2:
        st.markdown("##### Taxa de Cancelamento por Estado")
        df_cancel = df_f.copy()
        df_cancel['is_cancel'] = (
            df_cancel["status_simplificado"] == "Cancelado").astype(int)

        df_cancel_estado = df_cancel.groupby(
            "uf_cliente")["is_cancel"].mean().reset_index()

        fig_cancel = px.bar(df_cancel_estado, x="uf_cliente",
                            y="is_cancel", text_auto='.1%')
        fig_cancel.update_yaxes(tickformat=".0%")
        fig_cancel.update_traces(marker_color=WARNING, textfont_color='white')
        fig_cancel.update_layout(
            xaxis_title=None, yaxis_title=None,
            font=dict(color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_cancel, use_container_width=True)

    st.divider()

    st.markdown("### ⚙️ KPIs por Categoria")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("##### Atraso Médio (Top 10)")
        df_sla = df_f[df_f["dias_atraso"] > 0].groupby("categoria_label")["dias_atraso"].mean(
        ).reset_index().sort_values("dias_atraso", ascending=False).head(10)

        fig_sla = px.bar(df_sla, x="dias_atraso",
                         y="categoria_label", orientation="h", text_auto='.1f')
        fig_sla.update_layout(yaxis=dict(autorange="reversed"))

        fig_sla.update_traces(marker_color=WARNING, textfont_color='white')
        fig_sla.update_layout(
            xaxis_title=None, yaxis_title=None,
            font=dict(color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_sla, use_container_width=True)

    with colB:
        st.markdown("##### Tempo de Processamento (Dias)")
        df_proc_cat = df_f[df_f["tempo_processamento"] >= 0].groupby("categoria_label")["tempo_processamento"].mean(
        ).reset_index().sort_values("tempo_processamento", ascending=False).head(10)

        fig_proc_cat = px.bar(df_proc_cat, x="tempo_processamento",
                              y="categoria_label", orientation="h", text_auto='.1f')
        fig_proc_cat.update_layout(yaxis=dict(autorange="reversed"))

        fig_proc_cat.update_traces(
            marker_color=PRIMARY, textfont_color='white')
        fig_proc_cat.update_layout(
            xaxis_title=None, yaxis_title=None,
            font=dict(color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_proc_cat, use_container_width=True)

# ------------------------------------------------------------------------------
# ABA 2: SIMULADOR DE PREVISÃO (VERSÃO FINAL COM REGRAS REGIONAIS)
# ------------------------------------------------------------------------------
with tab_previsao:
    st.markdown("### 🧠 Simulador de Risco de Atraso")
    st.info(
        "Utilize este formulário para simular um novo pedido e prever o risco de entrega.")

    if model is None:
        st.error(
            "Modelo não carregado. Verifique se o arquivo .pkl existe na pasta models.")
    else:
        TODOS_ESTADOS = sorted(['AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MT', 'MS',
                                'MG', 'PA', 'PB', 'PR', 'PE', 'PI', 'RJ', 'RN', 'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO'])

        # Garante que temos categorias para exibir
        lista_categorias = sorted(mapa_reverso_categorias.keys(
        )) if mapa_reverso_categorias else ["Outros"]

        with st.form("simulador_form"):
            st.markdown("#### Dados do Pedido")
            c1, c2, c3 = st.columns(3)

            with c1:
                origem = st.selectbox(
                    # BA default
                    "📍 Origem (Vendedor)", TODOS_ESTADOS, index=4)
                destino = st.selectbox(
                    # RJ default
                    "🏠 Destino (Cliente)", TODOS_ESTADOS, index=18)
                prazo = st.number_input(
                    "📅 Prazo Prometido (Dias)", min_value=1, max_value=90, value=7)

            with c2:
                cat_label = st.selectbox("📦 Categoria", lista_categorias)
                peso = st.number_input(
                    "⚖️ Peso (g)", min_value=10, max_value=100000, value=500, step=100)
                aprovacao = st.number_input(
                    "⏳ Tempo Aprovação (Dias)", min_value=0, max_value=30, value=0)

            with c3:
                st.write("📐 **Dimensões (cm)**")
                cc1, cc2, cc3 = st.columns(3)
                comp = cc1.number_input("Comp", 1, 200, 20)
                larg = cc2.number_input("Larg", 1, 200, 20)
                alt = cc3.number_input("Alt", 1, 200, 10)
                pickup = st.checkbox("Pickup em Loja?", value=False)

            st.write("")
            btn_calc = st.form_submit_button("🚀 Calcular Previsão")

        if btn_calc:
            try:
                # 1. Preparar Input para o Modelo
                # Se não encontrar a categoria, usa a própria string (fallback)
                cat_tecnica = mapa_reverso_categorias.get(cat_label, cat_label)

                vol = comp * larg * alt
                peso_cub = vol / 6000

                entrada = pd.DataFrame([{
                    'peso_cubado_kg': peso_cub,
                    'vol_cm3': vol,
                    'peso_produto_g': peso,
                    'tempo_aprovacao': aprovacao,
                    'prazo_prometido': prazo,
                    'uf_vendedor': origem,
                    'uf_cliente': destino,
                    'flag_pickup': 1 if pickup else 0,
                    'categoria_produto': cat_tecnica
                }])

                # 2. O Modelo Estatístico faz a previsão
                dias_pred_modelo = model.predict(entrada)[0]

                # ==========================================================
                # 3. GUARDRAILS (Regras de Negócio e Lógica Física)
                # ==========================================================

                # Mapeamento de Macro-Regiões
                regioes_map = {
                    'AC': 'N', 'AL': 'NE', 'AP': 'N', 'AM': 'N', 'BA': 'NE',
                    'CE': 'NE', 'DF': 'CO', 'ES': 'SE', 'GO': 'CO', 'MA': 'NE',
                    'MT': 'CO', 'MS': 'CO', 'MG': 'SE', 'PA': 'N', 'PB': 'NE',
                    'PR': 'S', 'PE': 'NE', 'PI': 'NE', 'RJ': 'SE', 'RN': 'NE',
                    'RS': 'S', 'RO': 'N', 'RR': 'N', 'SC': 'S', 'SP': 'SE',
                    'SE': 'NE', 'TO': 'N'
                }

                reg_origem = regioes_map.get(origem, 'Outro')
                reg_destino = regioes_map.get(destino, 'Outro')

                # Definição de Tempo Mínimo de Transporte (SLA Físico)
                if origem == destino:
                    tempo_transporte_min = 1
                    tipo_rota = "Local"
                elif reg_origem == reg_destino:
                    tempo_transporte_min = 4
                    tipo_rota = "Regional"
                # Se envolve o Norte (AM, AP, etc)
                elif 'N' in [reg_origem, reg_destino]:
                    tempo_transporte_min = 12
                    tipo_rota = "Nacional (Difícil Acesso)"
                else:
                    tempo_transporte_min = 5
                    tipo_rota = "Nacional"

                # Cálculo do Tempo Total Mínimo Realista
                tempo_total_minimo = aprovacao + tempo_transporte_min
                atraso_fisico = tempo_total_minimo - prazo

                # A "Previsão Final" é o maior valor entre o que o modelo achou e a física
                dias_pred_final = max(dias_pred_modelo, atraso_fisico)

                # Verifica se houve intervenção da regra
                foi_ajustado = dias_pred_final > dias_pred_modelo

                # ==========================================================
                # 4. EXIBIÇÃO
                # ==========================================================
                st.divider()
                c_res1, c_res2 = st.columns([1, 2])

                total_dias_estimados = prazo + dias_pred_final

                with c_res1:
                    st.markdown("#### Resultado")
                    if dias_pred_final > 0:
                        st.error(f"⚠️ RISCO DE ATRASO")
                        st.metric("Atraso Estimado",
                                  f"+{dias_pred_final:.1f} dias")
                        if foi_ajustado:
                            st.caption(
                                f"ℹ️ Ajustado por regra logística ({origem}→{destino}).")
                    else:
                        st.success("✅ ENTREGA NO PRAZO")
                        st.metric("Margem", f"{abs(dias_pred_final):.1f} dias")

                with c_res2:
                    st.markdown("#### Diagnóstico Inteligente")

                    if dias_pred_final > 0:
                        analise_texto = (
                            f"O pedido é crítico. Além dos **{aprovacao} dias** de aprovação, "
                            f"a rota **{origem} ➝ {destino}** ({tipo_rota}) exige tempo de trânsito elevado. "
                            f"Estimativa total: **{total_dias_estimados:.1f} dias**."
                        )
                    else:
                        analise_texto = (
                            f"Operação segura. Rota **{tipo_rota}** com prazo confortável. "
                            f"Estimativa total: **{total_dias_estimados:.1f} dias**."
                        )

                    st.info(analise_texto)

                    # Gráfico de análise das etapas
                    dados_grafico = pd.DataFrame({
                        'Etapa': ['Aprovação', 'Transporte', 'Prazo Limite'],
                        'Dias': [aprovacao, max(0, total_dias_estimados - aprovacao), prazo],
                        'Cor': ['#808080', PRIMARY, WARNING]
                    })

                    fig_breakdown = px.bar(
                        dados_grafico,
                        x='Dias', y='Etapa', orientation='h', text_auto='.1f',
                        color='Cor', color_discrete_map="identity"
                    )
                    fig_breakdown.update_layout(height=250, margin=dict(
                        l=0, r=0, t=0, b=0), showlegend=False)
                    st.plotly_chart(fig_breakdown, use_container_width=True)

            except Exception as e:
                st.error(f"Erro ao processar a previsão: {e}")
