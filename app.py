import streamlit as st
import pandas as pd
import plotly.express as px
import json
import joblib
import numpy as np
from utils.utils import enriquecer_dados_dash, map_cat_correcao

# ==============================================================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Olist Logistica Centro de Comando",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.image(
    "https://d3hw41hpah8tvx.cloudfront.net/images/logo_ecossistema_66f532e37b.svg",
    width=180
)

PRIMARY = "#202652"
WARNING = "#ff4b4b"

st.markdown(f"""
<style>
div[data-testid="metric-container"] {{
    background-color: #ffffff;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid {PRIMARY};
    box-shadow: 2px 2px 5px rgba(0,0,0,0.08);
}}
h1, h2, h3 {{ color: {PRIMARY}; }}
/* Força cor branca em textos de gráficos se houver conflito */
g.pointtext {{ fill: white !important; }}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. CARREGAMENTO DE DADOS E MODELO
# ==============================================================================
@st.cache_data
def load_data():
    # 1. Carrega os dados brutos
    df = pd.read_csv(
        "data/data_dashboard_processed.csv",
        parse_dates=["data_aprovacao", "data_postagem"]
    )

    # 2. Aplica o tratamento centralizado (utils)
    df = enriquecer_dados_dash(df)

    return df


@st.cache_resource
def load_model():
    return joblib.load('models/modelo_previsao_atraso_olist.pkl')


df = load_data()
model = load_model()

# ==============================================================================
# 3. PREPARAÇÃO AUXILIAR
# ==============================================================================
if not df.empty and 'categoria_label' in df.columns:
    mapa_reverso_categorias = dict(
        zip(df['categoria_label'], df['categoria_produto']))
else:
    mapa_reverso_categorias = {
        cat: cat for cat in df['categoria_produto'].unique()}

# ==============================================================================
# 4. SIDEBAR
# ==============================================================================
st.sidebar.header("Filtros Globais")

status_options = sorted(df["status_simplificado"].astype(str).unique())
status_sel = st.sidebar.multiselect("Status do Pedido", status_options)

estados_sel = st.sidebar.multiselect(
    "Estado do Cliente", sorted(df["uf_cliente"].unique()))

cat_options = sorted(df["categoria_label"].unique())
cat_sel = st.sidebar.multiselect("Categoria de Produto", cat_options)

# Aplicando Filtros
df_f = df.copy()

if status_sel:
    df_f = df_f[df_f["status_simplificado"].isin(status_sel)]
if estados_sel:
    df_f = df_f[df_f["uf_cliente"].isin(estados_sel)]
if cat_sel:
    df_f = df_f[df_f["categoria_label"].isin(cat_sel)]


def abreviar(valor):
    if valor >= 1_000_000:
        return f"{valor/1_000_000:.1f}M"
    if valor >= 1_000:
        return f"{valor/1_000:.1f}K"
    return f"{valor:.0f}"


# ==============================================================================
# 5. ABAS DA APLICAÇÃO
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
                "Em Processamento": "#fcca00",
                "Cancelado": WARNING,
            }
        )
        fig_status.update_layout(
            legend=dict(orientation="h", yanchor="top",
                        y=-0.25, xanchor="center", x=0.5),
            margin=dict(t=60, b=60, l=10, r=10),
            height=300,
            font=dict(color='white'),  # Título e legendas brancos
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig_status.update_traces(textfont_color='white')  # Rótulos brancos
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
            xaxis_title=None, yaxis_title=None,  # Remove legendas X/Y
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
# ABA 2: SIMULADOR DE PREVISÃO
# ------------------------------------------------------------------------------
with tab_previsao:
    st.markdown("### 🧠 Simulador de Risco de Atraso")
    st.info("Preencha os dados abaixo para simular o risco de um novo pedido.")

    TODOS_ESTADOS = sorted(['AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MT', 'MS',
                            'MG', 'PA', 'PB', 'PR', 'PE', 'PI', 'RJ', 'RN', 'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO'])

    with st.form("simulador"):
        c1, c2, c3 = st.columns(3)

        with c1:
            origem = st.selectbox("📍 Origem", TODOS_ESTADOS, index=24)
            destino = st.selectbox("🏠 Destino", TODOS_ESTADOS, index=18)
            prazo = st.number_input("📅 Prazo (Dias)", 1, 90, 7)

        with c2:
            cat_label = st.selectbox("📦 Categoria", sorted(
                mapa_reverso_categorias.keys()))
            peso = st.number_input("⚖️ Peso (g)", 10, 30000, 500)
            aprovacao = st.number_input("⏳ Aprov. (Dias)", 0, 15, 0)

        with c3:
            st.write("📐 **Dimensões (cm)**")
            comp = st.number_input("Comprimento", 1, 200, 20)
            larg = st.number_input("Largura", 1, 200, 20)
            alt = st.number_input("Altura", 1, 200, 10)
            pickup = st.checkbox("Pickup?")

        btn_calc = st.form_submit_button("🔮 Calcular Previsão")

    if btn_calc:
        cat_tecnica = mapa_reverso_categorias[cat_label]

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

        try:
            dias_pred = model.predict(entrada)[0]

            st.divider()
            c_res1, c_res2 = st.columns([1, 2])

            with c_res1:
                if dias_pred > 0:
                    st.error("⚠️ Risco de Atraso")
                    st.metric("Atraso Estimado", f"+{dias_pred:.1f} dias")
                else:
                    st.success("✅ Entrega no Prazo")
                    st.metric("Margem", f"{abs(dias_pred):.1f} dias")

            with c_res2:
                total_dias = prazo + dias_pred
                st.markdown(
                    f"**Análise:** O pedido deve chegar em **{total_dias:.1f} dias** totais.")

                if dias_pred > 0:
                    st.warning("Recomendação: Aumente o prazo prometido.")
                else:
                    st.info("Operação segura.")

        except Exception as e:
            st.error(f"Erro ao processar previsão: {e}")
