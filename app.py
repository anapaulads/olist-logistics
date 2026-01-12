import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ==============================================================================
# 1. CONFIGURAÇÃO E ESTILO (CORRIGIDO)
# ==============================================================================
st.set_page_config(
    page_title="Olist Logistics Control Tower",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paleta de Cores Olist
COLOR_PRIMARY = '#202652'  # Azul Marinho
COLOR_SEC = '#fcce00'      # Amarelo
COLOR_BG = '#f0f2f6'

st.markdown(f"""
    <style>
    /* Ajuste de Fundo da Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {COLOR_BG};
    }}
    
    /* CORREÇÃO DOS CARDS (KPIs): 
       Forçamos a cor do texto para PRETO (#333) para garantir leitura no fundo branco 
    */
    div[data-testid="metric-container"] {{
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid {COLOR_PRIMARY};
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }}
    
    /* Força cor do rótulo e do valor para ficarem visíveis */
    div[data-testid="metric-container"] > label {{
        color: #333333 !important; 
    }}
    div[data-testid="metric-container"] > div {{
        color: {COLOR_PRIMARY} !important;
    }}
    
    /* Títulos */
    h1, h2, h3 {{
        color: {COLOR_PRIMARY};
    }}
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CARGA DE DADOS
# ==============================================================================


@st.cache_resource
def load_model():
    return joblib.load('models/modelo_previsao_atraso_olist.pkl')


@st.cache_data
def load_data():
    try:
        # Carrega o CSV que você separou para o Dashboard (com nomes legíveis)
        return pd.read_csv('data/data_dashboard_processed.csv')
    except FileNotFoundError:
        return None


model = load_model()
df = load_data()

# Lista Completa de UFs do Brasil
TODOS_ESTADOS = sorted([
    'AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MT', 'MS', 'MG',
    'PA', 'PB', 'PR', 'PE', 'PI', 'RJ', 'RN', 'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO'
])

# ==============================================================================
# 3. SIDEBAR (FILTROS)
# ==============================================================================
try:
    st.sidebar.image(
        "https://d3hw41hpah8tvx.cloudfront.net/images/logo_ecossistema_66f532e37b.svg", width=180)
except:
    st.sidebar.markdown("### 🚚 Olist Logistics")

st.sidebar.markdown("---")
st.sidebar.header("Filtros Globais")

if df is not None:
    df_filtered = df.copy()

    # Filtro de Região (Estado Cliente)
    sel_estado = st.sidebar.multiselect(
        "Filtrar Estado (Destino):", TODOS_ESTADOS)

    # Filtro de Categoria
    categorias = sorted(df['categoria_produto'].astype(str).unique())
    sel_cat = st.sidebar.multiselect("Filtrar Categoria:", categorias)

    if sel_estado:
        df_filtered = df_filtered[df_filtered['uf_cliente'].isin(sel_estado)]
    if sel_cat:
        df_filtered = df_filtered[df_filtered['categoria_produto'].isin(
            sel_cat)]
else:
    df_filtered = pd.DataFrame()

# ==============================================================================
# 4. DASHBOARD E SIMULADOR
# ==============================================================================
tab1, tab2 = st.tabs(["📊 Visão de Negócio (BI)", "🤖 Simulador Preditivo (IA)"])

# --- ABA 1: BUSINESS INTELLIGENCE ---
with tab1:
    if df is None:
        st.error(
            "⚠️ Erro: Arquivo 'tabela_dashboard.csv' não encontrado na pasta 'data'.")
    else:
        st.markdown("### 📊 Visão Geral da Operação")

        # --- LINHA 1: KPIs GERAIS ---
        col1, col2, col3, col4 = st.columns(4)

        qtd_pedidos = len(df_filtered)

        # Frete Médio (Se existir a coluna)
        col_frete = 'valor_frete' if 'valor_frete' in df_filtered.columns else 'freight_value'
        frete_medio = df_filtered[col_frete].mean(
        ) if col_frete in df_filtered.columns else 0

        # Taxa de Atraso
        atrasados = df_filtered[df_filtered['dias_atraso'] > 0]
        taxa_atraso = (len(atrasados) / qtd_pedidos *
                       100) if qtd_pedidos > 0 else 0

        # Taxa de Cancelamento (Se existir status)
        col_status = 'status_pedido' if 'status_pedido' in df_filtered.columns else 'order_status'
        if col_status in df_filtered.columns:
            cancelados = df_filtered[df_filtered[col_status] == 'canceled']
            taxa_cancel = (len(cancelados) / qtd_pedidos *
                           100) if qtd_pedidos > 0 else 0
        else:
            taxa_cancel = 0

        col1.metric("📦 Total de Pedidos", f"{qtd_pedidos:,.0f}")
        col2.metric("💰 Frete Médio", f"R$ {frete_medio:.2f}")
        col3.metric("⚠️ Taxa de Atraso", f"{taxa_atraso:.1f}%")
        col4.metric("🚫 Taxa Cancelamento",
                    f"{taxa_cancel:.1f}%", delta_color="inverse")

        st.markdown("---")

        # --- LINHA 2: GRÁFICOS DE NEGÓCIO ---
        c_chart1, c_chart2 = st.columns(2)

        with c_chart1:
            st.markdown("##### 💵 Custo de Frete por Estado (Top 10)")
            if col_frete in df_filtered.columns:
                df_frete = df_filtered.groupby('uf_cliente')[col_frete].mean(
                ).reset_index().sort_values(col_frete, ascending=False).head(10)

                fig_frete = px.bar(
                    df_frete,
                    x='uf_cliente',
                    y=col_frete,
                    text_auto='.2f',
                    title="Onde o Frete é mais caro?",
                    color=col_frete,
                    color_continuous_scale='Blues'  # Escala Azul
                )
                st.plotly_chart(fig_frete, use_container_width=True)
            else:
                st.info("Coluna de valor de frete não encontrada.")

        with c_chart2:
            st.markdown("##### 📦 Status dos Pedidos")
            if col_status in df_filtered.columns:
                df_status = df_filtered[col_status].value_counts(
                ).reset_index()
                df_status.columns = ['Status', 'Qtd']

                # Gráfico de Rosca
                fig_status = px.pie(
                    df_status,
                    values='Qtd',
                    names='Status',
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                st.plotly_chart(fig_status, use_container_width=True)
            else:
                st.info("Coluna de status não encontrada.")

        # --- LINHA 3: VISÃO DE PROBLEMAS (ATRASOS) ---
        st.markdown("### 🚨 Raio-X dos Atrasos")
        c_chart3, c_chart4 = st.columns(2)

        with c_chart3:
            st.markdown("##### Performance por Categoria (Piores)")
            df_atraso_cat = df_filtered[df_filtered['dias_atraso'] >
                                        0]['categoria_produto'].value_counts().head(10).reset_index()
            df_atraso_cat.columns = ['Categoria', 'Qtd Atrasos']

            fig_cat = px.bar(
                df_atraso_cat,
                x='Qtd Atrasos',
                y='Categoria',
                orientation='h',
                color_discrete_sequence=['#ff4b4b']  # Vermelho alerta
            )
            fig_cat.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_cat, use_container_width=True)

        with c_chart4:
            st.markdown("##### Mapa de Calor (Média de Atraso)")
            df_mapa = df_filtered.groupby('uf_cliente')[
                'dias_atraso'].mean().reset_index()
            fig_map = px.choropleth(
                df_mapa,
                geojson="https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson",
                featureidkey="properties.sigla",
                locations="uf_cliente",
                color="dias_atraso",
                color_continuous_scale="Reds",
                scope="south america",
                center={"lat": -14.2350, "lon": -51.9253}
            )
            fig_map.update_geos(fitbounds="locations", visible=False)
            st.plotly_chart(fig_map, use_container_width=True)

# --- ABA 2: SIMULADOR ---
with tab2:
    st.markdown("### 🤖 Simulador Preditivo")
    st.info("Este módulo utiliza Inteligência Artificial para prever riscos em novos pedidos antes que eles aconteçam.")

    with st.form("simulador"):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            origem = st.selectbox("📍 Origem (Vendedor)",
                                  TODOS_ESTADOS, index=24)  # Default SP
            destino = st.selectbox("🏠 Destino (Cliente)",
                                   TODOS_ESTADOS, index=18)  # Default RJ
            prazo = st.number_input("📅 Prazo Prometido (Dias)", 1, 90, 7)

        with col_b:
            lista_cats = sorted(df['categoria_produto'].astype(
                str).unique()) if df is not None else ['outros']
            cat = st.selectbox("📦 Categoria", lista_cats)
            peso = st.number_input("⚖️ Peso (gramas)", 10, 30000, 500)
            aprovacao = st.number_input("⏳ Tempo Aprovação (Dias)", 0, 15, 0)

        with col_c:
            st.write("📐 **Dimensões (cm)**")
            comp = st.number_input("Comprimento", 1, 200, 20)
            larg = st.number_input("Largura", 1, 200, 20)
            alt = st.number_input("Altura", 1, 200, 10)
            pickup = st.checkbox("Retirada em Loja?")

        btn_calc = st.form_submit_button("🔮 Calcular Previsão")

    if btn_calc:
        # 1. Preparar Dados
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
            'categoria_produto': cat
        }])

        # 2. Previsão
        try:
            dias_pred = model.predict(entrada)[0]

            st.divider()
            c_res1, c_res2 = st.columns([1, 2])

            with c_res1:
                if dias_pred > 0:
                    st.error("⚠️ Risco de Atraso")
                    st.metric("Estimativa", f"+{dias_pred:.1f} dias")
                else:
                    st.success("✅ Dentro do Prazo")
                    st.metric("Folga Estimada", f"{abs(dias_pred):.1f} dias")

            with c_res2:
                total = prazo + dias_pred
                st.markdown(
                    f"**Análise:** O produto deve chegar em **{total:.1f} dias** totais.")
                if dias_pred > 0:
                    st.warning(
                        "Recomendação: Revise o prazo prometido ou utilize envio expresso.")
                else:
                    st.info("Recomendação: Operação segura. Risco baixo.")

        except Exception as e:
            st.error(f"Erro no modelo: {e}")
