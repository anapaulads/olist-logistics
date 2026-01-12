import streamlit as st
import pandas as pd
import plotly.express as px
import json

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

st.markdown("""
<style>
div[data-testid="metric-container"] {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #202652;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.08);
}
h1, h2, h3 { color: #202652; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    return pd.read_csv(
        "data/data_dashboard_processed.csv",
        parse_dates=["data_aprovacao", "data_postagem"]
    )


df = load_data()

df["faturamento_pedido"] = df["preco_produto"] + df["valor_frete"]

status_map = {
    "delivered": "Aprovado",
    "shipped": "Aprovado",
    "approved": "Em Processamento",
    "processing": "Em Processamento",
    "invoiced": "Em Processamento",
    "canceled": "Cancelado",
    "unavailable": "Cancelado"
}
df["status_simplificado"] = df["status_pedido"].map(status_map)

df["flag_atraso"] = df["dias_atraso"] > 0

df["tempo_processamento"] = (
    df["data_postagem"] - df["data_aprovacao"]
).dt.total_seconds() / 86400

st.sidebar.header("Filtros Globais")

status_options = (
    df["status_simplificado"]
    .dropna()
    .astype(str)
    .unique()
)

status_sel = st.sidebar.multiselect(
    "Status do Pedido",
    sorted(status_options)
)

estados_sel = st.sidebar.multiselect(
    "Estado do Cliente",
    sorted(df["uf_cliente"].unique())
)

df_f = df.copy()

if status_sel:
    df_f = df_f[df_f["status_simplificado"].isin(status_sel)]

if estados_sel:
    df_f = df_f[df_f["uf_cliente"].isin(estados_sel)]


def abreviar(valor):
    if valor >= 1_000_000:
        return f"{valor/1_000_000:.1f}M"
    if valor >= 1_000:
        return f"{valor/1_000:.1f}K"
    return f"{valor:.0f}"


st.markdown("## 📊 Visão Geral da Operação")

col1, col2, col3, col4, col5 = st.columns(5)

total_pedidos = len(df_f)
faturamento = df_f["faturamento_pedido"].sum()
taxa_atraso = df_f["flag_atraso"].mean() * 100
atraso_medio = df_f.loc[df_f["flag_atraso"], "dias_atraso"].mean()
taxa_cancel = (df_f["status_simplificado"] == "Cancelado").mean() * 100

col1.metric("📦 Pedidos", abreviar(total_pedidos))
col2.metric("💰 Faturamento", f"R$ {abreviar(faturamento)}")
col3.metric("🚨 Taxa de Atraso", f"{taxa_atraso:.1f}%")
col4.metric("⏱️ Atraso Médio",
            f"{0 if pd.isna(atraso_medio) else atraso_medio:.1f} dias")
col5.metric("🚫 Cancelamento", f"{taxa_cancel:.1f}%")

st.divider()

colA, colB = st.columns([1, 1.5])

with colA:
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
            "Cancelado": WARNING
        }
    )

    fig_status.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=60, b=60, l=10, r=10),
        height=300
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(fig_status, use_container_width=False,
                    width=420, height=300)

with colB:
    with open("data/brazil_states.geojson", "r", encoding="utf-8") as f:
        brazil_geojson = json.load(f)

    df_geo = (
        df_f[df_f["dias_atraso"] > 0]
        .groupby("uf_cliente")
        .agg(
            pedidos=("pedido_id", "count"),
            atraso_medio=("dias_atraso", "mean")
        )
        .reset_index()
    )

    fig_map = px.choropleth(
        df_geo,
        geojson=brazil_geojson,
        locations="uf_cliente",
        featureidkey="properties.sigla",
        color="atraso_medio",
        color_continuous_scale="Reds"
    )

    fig_map.update_geos(fitbounds="locations", visible=False)
    fig_map.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=60, b=0),
        height=300
    )

    st.plotly_chart(fig_map, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    df_fat_estado = (
        df_f.groupby("uf_cliente")["faturamento_pedido"]
        .sum()
        .reset_index()
        .sort_values("faturamento_pedido", ascending=False)
        .head(10)
    )

    fig_fat = px.bar(
        df_fat_estado,
        x="uf_cliente",
        y="faturamento_pedido",
        title="Top 10 Estados por Faturamento"
    )
    st.plotly_chart(fig_fat, use_container_width=True)

with col2:
    df_cancel_estado = (
        df_f.assign(cancel=lambda x: x["status_simplificado"] == "Cancelado")
        .groupby("uf_cliente")["cancel"]
        .mean()
        .reset_index()
    )

    fig_cancel = px.bar(
        df_cancel_estado,
        x="uf_cliente",
        y="cancel",
        title="Taxa de Cancelamento por Estado"
    )
    st.plotly_chart(fig_cancel, use_container_width=True)

st.divider()

st.markdown("### ⚙️ SLA — Atraso Médio por Categoria")

colA, colB = st.columns(2)

with colA:
    df_sla = (
        df_f[df_f["dias_atraso"] > 0]
        .groupby("categoria_produto")["dias_atraso"]
        .mean()
        .reset_index()
        .sort_values("dias_atraso", ascending=False)
        .head(10)
    )

    fig_sla = px.bar(
        df_sla,
        x="dias_atraso",
        y="categoria_produto",
        orientation="h",
        title="Atraso Médio por Categoria (dias)"
    )

    fig_sla.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_sla, use_container_width=True)

with colB:
    df_proc_cat = (
        df_f[
            df_f["tempo_processamento"].notna() &
            (df_f["tempo_processamento"] >= 0)
        ]
        .groupby("categoria_produto")["tempo_processamento"]
        .mean()
        .reset_index()
        .sort_values("tempo_processamento", ascending=False)
        .head(10)
    )

    fig_proc_cat = px.bar(
        df_proc_cat,
        x="tempo_processamento",
        y="categoria_produto",
        orientation="h",
        title="Tempo Médio de Processamento por Categoria (dias)"
    )

    st.plotly_chart(fig_proc_cat, use_container_width=True)
