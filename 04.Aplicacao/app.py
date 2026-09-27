import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
import joblib
from pathlib import Path
# Configuração da página
st.set_page_config(page_title="Simulador de Precificação B2B", layout="wide")
st.title("🎯 Simulador B2B de Precificação e Risco (Dafiti)")

# ---------------------------------------------------------
# 1. CARREGAMENTO DOS MODELOS SALVOS (COM CACHE)
# ---------------------------------------------------------
@st.cache_resource
def carregar_modelos():
    pasta_atual = Path(__file__).resolve().parent
    caminho_modelo = pasta_atual.parent / "02.Modelo" / "artefatos_precificacao.joblib"

    if not caminho_modelo.exists():
        return None

    return joblib.load(caminho_modelo)

artefatos = carregar_modelos()

if artefatos is None:
    st.error("⚠️ Arquivo de modelos não encontrado. Rode o script de treinamento primeiro para gerar o 'artefatos_precificacao.joblib'.")
    st.stop()

modelo_media = artefatos["modelo_media"]
modelo_variancia = artefatos["modelo_variancia"]
colunas_X = artefatos["colunas_X"]
q_adapt = artefatos["q_adapt"]

# ---------------------------------------------------------
# 2. INTERFACE DE ENTRADA DO USUÁRIO (BARRA LATERAL)
# ---------------------------------------------------------
st.sidebar.header("⚙️ Parâmetros do Produto")

preco_original = st.sidebar.number_input("Preço Original (R$)", min_value=10.0, max_value=5000.0, value=300.0, step=10.0)
parcelas = st.sidebar.selectbox("Quantidade de Parcelas", options=[1, 2, 3, 4, 5])

todas_tags = [col.replace('Tag_', '') for col in colunas_X if col.startswith('Tag_')]
tags_selecionadas = st.sidebar.multiselect("Características (Tags)", options=todas_tags, default=['Tênis', 'Calçados Femininos'])

# ---------------------------------------------------------
# 3. CONSTRUÇÃO DAS ABAS DA APLICAÇÃO
# ---------------------------------------------------------
aba1, aba2 = st.tabs(["📊 Simulador", "📖 Sobre a Ferramenta"])

with aba1:
    st.markdown("Insira as características do lote no menu lateral para obter a faixa ideal de desconto e a análise de incerteza do mercado.")
    
    # Montagem do Input (Vetor de características)
    input_df = pd.DataFrame(0, index=[0], columns=colunas_X)
    input_df.at[0, 'Preço Original Numérico'] = preco_original
    input_df.at[0, 'Preço Parcelado Numérico'] = preco_original 

    coluna_parcela = f"Quantidade de Parcelas_{parcelas}x"
    if coluna_parcela in colunas_X:
        input_df.at[0, coluna_parcela] = 1

    for tag in tags_selecionadas:
        coluna_tag = f"Tag_{tag}"
        if coluna_tag in colunas_X:
            input_df.at[0, coluna_tag] = 1

    # Predição Rápida
    pred_desconto = modelo_media.predict(input_df)[0]
    variancia = np.maximum(modelo_variancia.predict(input_df)[0], 0)
    escala = np.sqrt(variancia)

    limite_inferior = pred_desconto - (q_adapt * escala)
    limite_superior = pred_desconto + (q_adapt * escala)

    # Exibição dos KPIs
    st.subheader("📊 Recomendação de Posicionamento")
    col1, col2, col3 = st.columns(3)
    col1.metric("Desconto Médio Previsto", f"R$ {pred_desconto:.2f}")
    col2.metric("Piso (Margem Inferior)", f"R$ {limite_inferior:.2f}", delta="Risco Alto de Encalhe", delta_color="inverse")
    col3.metric("Teto (Margem Superior)", f"R$ {limite_superior:.2f}", delta="Risco Alto de Prejuízo", delta_color="off")

    st.info("""
    💡 **Leitura B2B:** Posicione o desconto do seu produto entre o Piso e o Teto. Valores fora dessa faixa são considerados estatisticamente anômalos no mercado atual.

    **Entendendo os Riscos (Glossário da Margem):**
    * 📉 **Risco de Encalhe (Abaixo do Piso):** Oferecer um desconto menor do que o mínimo recomendado faz com que seu produto perca competitividade. Ele ficará caro demais em relação à média do site, travando o seu estoque.
    * 💸 **Risco de Prejuízo (Acima do Teto):** Oferecer um desconto maior do que o teto recomendado significa que você está "deixando dinheiro na mesa". O modelo indica que o mercado aceitaria comprar por um preço mais alto, logo, dar tanto desconto destrói a sua margem de lucro sem necessidade.
    
    ---
    **🤖 Por que a predição pontual não é uma subtração exata (Original - Pix)?**
    O motor analítico desta ferramenta é um *Random Forest Regressor*, um modelo não-linear de *ensemble learning*. Ele não opera resolvendo uma equação aritmética determinística ($y = a - b$). Em vez disso, o algoritmo mapeia as variáveis de entrada em um espaço multidimensional (*feature space*) e estima a esperança condicional do desconto ($E[Y|X]$). Portanto, o valor sugerido representa a convergência estatística dos descontos historicamente praticados e absorvidos pelo mercado para partições de produtos com esse exato vetor de características (faixa de preço, tags e parcelamento).
    """)

    # SHAP
    st.markdown("---")
    st.subheader("🔍 Por que o modelo sugeriu este intervalo?")

    explainer_variancia = shap.TreeExplainer(modelo_variancia)
    shap_values = explainer_variancia(input_df)

    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title("Fatores que aumentam ou reduzem a Incerteza/Risco desta precificação")
    plt.tight_layout()
    st.pyplot(fig)

with aba2:
    st.markdown("""
    ### 🎯 O Intuito da Aplicação
    O Simulador de Precificação B2B foi desenvolvido para auxiliar lojistas e marcas a definirem descontos estratégicos de forma orientada a dados no e-commerce. O objetivo central é maximizar a lucratividade encontrando o "ponto ótimo" de preço, protegendo o vendedor contra dois grandes riscos: o **encalhe de estoque** (quando o desconto é muito baixo e o produto perde competitividade) e o **prejuízo financeiro** (quando o desconto oferecido é maior do que o mercado exigiria, destruindo a margem de lucro).

    ### ⚙️ Como Funciona
    A ferramenta utiliza algoritmos de *Machine Learning* aliados a uma técnica avançada chamada **Conformal Prediction** (Predição Conformal) para analisar o cenário:

    1. **Entrada de Dados:** O usuário insere as características do lote, como preço original, preço Pix, parcelamento e categoria do calçado utilizando o menu lateral.
    2. **Cálculo da Faixa Segura:** Em vez de dar apenas um "chute" de preço, a IA calcula dinamicamente um **Piso** e um **Teto** recomendados. O tamanho dessa faixa se adapta à dificuldade de prever cada item (ex: sapatos de luxo geram mais incerteza estatística e, portanto, recebem margens de segurança maiores).
    3. **Interpretabilidade (SHAP):** Para garantir transparência, o sistema gera um gráfico interativo explicando exatamente *quais fatores* (como o valor do produto ou sua categoria) estão influenciando o risco e a largura da margem sugerida pela inteligência artificial.
    """)