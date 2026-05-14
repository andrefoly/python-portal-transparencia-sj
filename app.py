import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression

# Configuração inicial da página
st.set_page_config(
    page_title="Portal Transparência SJBV - Multas",
    page_icon="📊",
    layout="wide"
)


# --- DADOS REAIS E SIMULADOS ---
@st.cache_data
def load_real_data():
    # Dados reais de arrecadação
    data = {
        "Referência": [
            "2022/JAN", "2022/FEV", "2022/MAR", "2022/ABR", "2022/MAI", "2022/JUN",
            "2023/JAN", "2023/FEV", "2023/MAR", "2023/ABR", "2023/MAI", "2023/JUN",
            "2024/JAN", "2024/FEV", "2024/MAR", "2024/ABR", "2024/MAI", "2024/JUN",
            "2025/JAN", "2025/FEV", "2025/MAR", "2025/ABR", "2025/MAI"
        ],
        "Arrecadação (R$)": [
            205633.57, 101722.49, 99291.17, 73657.07, 118405.07, 98839.71,
            148084.06, 59323.34, 66970.74, 50022.36, 75248.06, 88244.94,
            164773.49, 76644.55, 131295.85, 158259.74, 201245.06, 175949.36,
            386506.52, 196676.28, 150534.01, 152427.99, 152345.50
        ],
        "Qtd_Infrações": [
            1038, 486, 522, 356, 564, 458,
            672, 270, 300, 222, 372, 399,
            680, 316, 642, 804, 1000, 864,
            1673, 916, 678, 643, 733
        ]
    }

    df_multas = pd.DataFrame(data)

    # --- SIMULAÇÃO DE INVESTIMENTOS POR SETOR ---
    # Aqui adicionamos os setores solicitados e valores simulados
    data_investimentos = {
        "Setor": [
            "Saneamento", "Recapeamento/Ruas", "Iluminação Pública",
            "Educação no Trânsito", "Sinalização Viária", "Segurança/Policiamento"
        ],
        "Valor Investido (R$)": [850000.00, 1200000.00, 450000.00, 250000.00, 320000.00, 180000.00]
    }

    return df_multas, pd.DataFrame(data_investimentos)


df_multas, df_investimentos = load_real_data()


# --- LÓGICA DO MODELO PREDITIVO ---
def extrair_insights_preditivos(df):
    X = np.arange(len(df)).reshape(-1, 1)
    y = df["Arrecadação (R$)"].values
    model = LinearRegression().fit(X, y)

    proximo_mes_idx = np.array([[len(df)]])
    predicao = model.predict(proximo_mes_idx)[0]
    ultimo_valor = y[-1]
    variacao = ((predicao - ultimo_valor) / ultimo_valor) * 100
    return predicao, variacao


previsao_valor, variacao_percent = extrair_insights_preditivos(df_multas)

# --- INTERFACE DO DASHBOARD ---
st.title("🏙️ Portal de Transparência Ativa - São João da Boa Vista")
st.subheader("Análise de Arrecadação e Destinação de Recursos")
st.markdown("*Dados Reais (Multas) e Simulados (Investimentos por Setor)*")

st.divider()

# KPIs Principais
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Arrecadado", f"R$ {df_multas['Arrecadação (R$)'].sum():,.2f}")
with col2:
    st.metric("Total de Infrações", f"{df_multas['Qtd_Infrações'].sum():,}")
with col3:
    st.metric("Tendência Próximo Mês", f"R$ {previsao_valor:,.2f}", f"{variacao_percent:.1f}%")
with col4:
    total_investido = df_investimentos["Valor Investido (R$)"].sum()
    st.metric("Total Investido na Cidade", f"R$ {total_investido:,.2f}")

st.divider()

# --- ÁREA DE GRÁFICOS ---
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### 📈 Histórico e Tendência de Arrecadação")
    X_seq = np.arange(len(df_multas)).reshape(-1, 1)
    modelo = LinearRegression().fit(X_seq, df_multas["Arrecadação (R$)"])
    df_multas["Tendência"] = modelo.predict(X_seq)

    fig_trend = px.line(
        df_multas,
        x="Referência",
        y=["Arrecadação (R$)", "Tendência"],
        markers=True,
        color_discrete_map={"Arrecadação (R$)": "#00CC96", "Tendência": "#EF553B"}
    )
    fig_trend.update_layout(legend_title_text='Legenda', margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_trend, use_container_width=True)

with col_right:
    st.markdown("### 🏗️ Distribuição de Investimentos por Setor")
    # Gráfico de Rosca para os Setores
    fig_setores = px.pie(
        df_investimentos,
        values='Valor Investido (R$)',
        names='Setor',
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_setores.update_traces(textinfo='percent+label')
    fig_setores.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_setores, use_container_width=True)

# --- TABELAS E INSIGHTS ---
st.divider()
tab1, tab2, tab3 = st.tabs(["📊 Dados de Multas", "👷 Detalhes de Investimento", "🤖 Insights da IA"])

with tab1:
    st.dataframe(df_multas[["Referência", "Arrecadação (R$)", "Qtd_Infrações"]], use_container_width=True)

with tab2:
    # Mostra a tabela de investimentos com barra de progresso visual
    st.dataframe(
        df_investimentos.style.format({"Valor Investido (R$)": "R$ {:,.2f}"})
        .bar(subset=["Valor Investido (R$)"], color='#636EFA'),
        use_container_width=True
    )

with tab3:
    status_tendencia = "aumento" if variacao_percent > 0 else "queda"
    maior_setor = df_investimentos.loc[df_investimentos['Valor Investido (R$)'].idxmax(), 'Setor']

    st.info(f"""
        💡 **Análise de Transparência:**
        * O modelo identifica uma tendência de **{status_tendencia} de {abs(variacao_percent):.1f}%** na arrecadação.
        * Atualmente, o setor de **{maior_setor}** é o que recebe o maior volume de recursos simulados.
        * A integração destes dados permite ao cidadão visualizar como o valor das multas retorna em melhorias para a cidade.
    """
            )

# Rodapé
st.sidebar.markdown(f"### {st.session_state.get('projeto_titulo', 'Projeto Integrador III')}")
st.sidebar.write(f"**Responsável:** Alcides André Muniz Foly")
st.sidebar.markdown("---")
st.sidebar.caption("Ferramentas: Streamlit, Plotly, Scikit-Learn")
