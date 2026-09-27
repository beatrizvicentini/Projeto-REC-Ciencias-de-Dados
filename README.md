# Pipeline Analítico de Precificação Dinâmica e Quantificação de Risco

Este repositório consolida um projeto de ponta a ponta voltado à inteligência de precificação para o varejo de e-commerce. O fluxo abrange desde a coleta autônoma de dados em larga escala até o desenvolvimento de um aplicativo interativo B2B que utiliza *Machine Learning* para sugerir descontos e quantificar incertezas mercadológicas.

## 1. Coleta de Dados (Web Scraping)

A base de dados foi construída através de um processo de extração automatizada do catálogo de calçados (Feminino e Masculino) da plataforma Dafiti.

* **Ferramentas:** `requests` e `BeautifulSoup4`.
* **Evasão de Bloqueios (Anti-Bot/WAF):** Para contornar os sistemas de segurança do e-commerce e evitar bloqueios de IP durante a paginação extensiva, o script implementa uma arquitetura de navegação persistente (`requests.Session()`) para reter *cookies* de sessão.
* **Mimetização Humana:** Foram injetados *headers* completos de navegadores reais (incluindo `User-Agent`, `Accept-Language` e `Referer`) e aplicadas pausas estocásticas (`random.uniform`) entre as requisições, descaracterizando o padrão temporal robótico.
* **Saída:** Os dados brutos consolidados contemplam atributos como Preço Original, Preço Pix, parcelamento, marca e tags de categorização multidimensional, exportados em formato `.csv`.

## 2. Modelagem Preditiva e Conformal Prediction

O núcleo analítico do projeto visa prever o desconto ideal (`Desconto Pix`) não apenas como um valor pontual, mas como uma faixa estatisticamente calibrada utilizando a metodologia de **Conformal Prediction**.

* **Pré-processamento:** Limpeza de *strings* financeiras e aplicação de *one-hot encoding* multirrótulo nas tags de produtos.
* **Modelo Base:** Um algoritmo não-linear `RandomForestRegressor` foi treinado para mapear o espaço de *features* e prever a média condicional do desconto.
* **Conformal Prediction Fixo:** Utilizando um conjunto de calibração isolado, os resíduos (erros absolutos) do modelo foram avaliados para calcular um quantil estático de 90% de cobertura, gerando margens de erro idênticas para todos os produtos.
* **Conformal Prediction Adaptativo (Variância):** Para lidar com a heteroscedasticidade dos dados (variância que aumenta conforme o valor do produto), um **segundo modelo** `RandomForestRegressor` foi treinado especificamente para prever os *resíduos ao quadrado* (o tamanho do erro) do primeiro modelo. Isso permite que a faixa de incerteza seja dinâmica: margens curtas para calçados previsíveis (baratos) e "paraquedas" estatísticos largos para sapatos de luxo, mantendo a cobertura global de 90%.

## 3. Interpretabilidade Analítica (SHAP)

Para quebrar o aspecto "caixa preta" dos modelos *ensemble*, aplicamos o *SHapley Additive exPlanations* (SHAP) em duas frentes distintas de análise de negócio:

* **SHAP na Média (Previsão Pontual):** Revelou que a variável `Preço Original Numérico` atua como o motor primário do desconto predito. Variáveis categóricas (como o tipo do sapato) possuem impacto quase nulo na regressão direta.
* **SHAP na Variância (Incerteza):** Mapeou os gatilhos de risco do modelo. Demonstrou categoricamente que valores extremos de calçados injetam imensa incerteza no sistema. *Waterfall plots* individuais confirmaram que as margens alargadas são acionadas pela dificuldade do algoritmo em ancorar o comportamento do mercado em itens de altíssimo valor agregado (ticket alto).

## 4. Simulador B2B (Aplicação Streamlit)

Os artefatos de modelagem foram envelopados em uma aplicação *frontend* interativa, projetada para apoiar lojistas e analistas de *pricing* em tempo real.

* **Funcionalidade:** O usuário insere as características de um lote de calçados (preço, parcelamento, tags) na barra lateral. O sistema calcula simultaneamente a predição pontual e os limites adaptativos.
* **Decisão Estratégica:** A ferramenta entrega uma faixa de preço recomendada ancorada em dois KPIs de risco:
* **Piso (Risco de Encalhe):** Limite inferior da margem conformal. Precificar abaixo disso torna o produto caro frente à média do mercado.
* **Teto (Risco de Prejuízo):** Limite superior da margem conformal. Oferecer descontos acima deste teto significa corroer a margem de lucro em um cenário onde o mercado aceitaria pagar mais.


* **Transparência Matemática:** A aplicação esclarece que o resultado não é uma subtração exata (Preço - Pix), mas a esperança condicional estatística ($E[Y\vert{}X]$) baseada no histórico. Além disso, exibe um gráfico SHAP *Waterfall* em tempo real para justificar o alargamento ou estreitamento da margem recomendada na simulação atual.
---

## 5. Configuração e Execução do Ambiente

O projeto utiliza o gerenciador de pacotes e ambientes virtuais `uv`. A sincronização inicial das dependências (que incluem `pandas`, `scikit-learn`, `shap`, `matplotlib`, `jupyterlab` e `streamlit`) é feita na raiz do repositório executando:

### Instalar UV
Primeiro certifique-se que seu computador possui UV instalado. Basta seguir o link de instalacao em [documento instalacao UV](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer).


### Clonar o repositório e criar ambiente virtual
Abra o terminal na pasta onde quer o projeto e execute:

```bash
git clone https://github.com/beatrizvicentini/Projeto-REC-Ciencias-de-Dados.git Projeto-REC-Ciencias-de-Dados
cd Projeto-REC-Ciencias-de-Dados
uv init
```
Em seguida, prepare o ambiente com as bibliotecas necessarias:

```bash
uv add -r requirements.txt
```

Você pode escolher entre duas rotas para conduzir o fluxo do projeto:

### Caminho 1: Do Zero (Com Web Scraping Interativo)

1. **Inicialize o Jupyter Lab via console:**
```bash
uv run jupyter lab
```


2. No navegador que se abrirá, navegue até a pasta `01.Extracao de Dados` e abra o notebook de extração. Pressione **Run All** para executar a raspagem do catálogo da Dafiti.
3. Em seguida, acesse a pasta `02. Modelo`, abra o notebook de modelagem e clique em **Run All** para treinar os algoritmos e gerar os artefatos conformal.
4. Por fim, na pasta `04.Aplicacao`, inicie o simulador interativo:
```bash
uv run streamlit run "04.Aplicacao/app.py"
```



---

### Caminho 2: Rota Rápida (Com Base Pré-coletada)

1. Caso queira pular a etapa de web scraping, abra o Jupyter Lab a partir da raiz:
```bash
uv run jupyter lab
```


2. Abra o notebook de download **baixar_dados** localizado na raiz e clique em **Run All** para obter a base consolidada.
2.1. Alternativa manual: se preferir, acesse [esta pasta no Drive](https://drive.google.com/drive/folders/1ELvahzSeIOTHKbfjhb57MJiYh3JX97Hq?usp=sharing), baixe tudo e extraia para 00.Dados/ na raiz do projeto.
3. Abra o notebook na pasta `02.Modelo` e execute **Run All** para processar os dados e salvar os artefatos.
4. Inicie o servidor do simulador B2B no terminal:
```bash
uv run streamlit run "04.Aplicacao/app.py"
```



O servidor web do Streamlit será iniciado localmente e a interface do simulador estará disponível no navegador através do endereço `


### 6. Relatório e Análise dos Modelos

O projeto conta com uma documentação analítica detalhada localizada na pasta `03.Relatorio`.

* **Conteúdo:** A pasta abriga um notebook Jupyter contendo a exploração completa dos dados, a avaliação estatística detalhada das predições e a interpretação visual dos gráficos SHAP gerados para os modelos de média e variância.
* **Como visualizar:** Você pode abrir o relatório interativamente executando o Jupyter Lab a partir da raiz do repositório:
```bash
uv run jupyter lab
```


Em seguida, navegue até a pasta `03.Relatorio` e abra o notebook correspondente para acompanhar a exploração dos resultados e métricas.