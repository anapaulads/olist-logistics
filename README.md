# 🚚 Olist Logistica Centro de Comando (End-to-End Data Science Project)

![Status](https://img.shields.io/badge/Status-Concluído-green) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-App-red) ![Machine Learning](https://img.shields.io/badge/Model-RandomForest-orange)

> 🚀 **Destaques do Projeto:** Este portfólio demonstra domínio em **Full-Stack Data Science**: da engenharia de dados (ETL robusto e Feature Engineering) à construção de pipelines de **Machine Learning** e **Deploy** de aplicações web. Evidencia forte capacidade analítica em **Supply Chain & Logística**, aliada a boas práticas de **Engenharia de Software** (modularização, código limpo e controle de versão), provando aptidão para resolver problemas de negócio complexos de ponta a ponta.

## 💼 Contexto e Problema de Negócio
A **Olist** atua como uma grande loja de departamentos dentro de marketplaces, conectando pequenas empresas a clientes finais. Nesse modelo, a logística é descentralizada e complexa.

**O Problema:** Atrasos na entrega são a principal causa de insatisfação (Churn) e custos operacionais (Reclamações/SAC).
**A Solução:** Uma "Torre de Controle Logístico" composta por:
1.  **Dashboard Analítico:** Para monitoramento de KPIs em tempo real (Loss Rate, Atraso Médio, Faturamento).
2.  **Motor de Previsão (IA):** Um modelo preditivo que estima o risco de atraso *antes* da compra ser finalizada, permitindo alinhar expectativas de prazo com o cliente.

---

## 🛠️ Pipeline do Projeto (Metodologia)
O projeto segue o ciclo de vida completo de Ciência de Dados (CRISP-DM):

### 1. Engenharia de Dados (ETL)
* **Fonte:** Dados públicos do E-commerce Brasileiro (Kaggle).
* **Limpeza Avançada:**
    * Tratamento cronológico: Remoção de inconsistências (ex: entregas registradas antes da compra).
    * Segmentação de Nulos: Diferenciação entre pedidos em andamento (WIP) e erros sistêmicos (Ruído).
* **Feature Engineering:** Criação de variáveis como `volume_cubico`, `tempo_aprovacao` e `densidade_rota`.

### 2. Análise Exploratória (Insights) 📊
Aprofundando nos dados, descobrimos padrões cruciais para a operação:
* **Desigualdade Regional:** Enquanto o Sudeste opera com prazos otimizados, regiões Norte e Nordeste apresentam SLA de entrega até **3x maior**, sugerindo a necessidade de CDs (Centros de Distribuição) locais.
* **O "Gargalo Invisível":** Pedidos com longo `tempo_aprovacao` (pagamento/análise de crédito) têm correlação direta com atrasos na entrega. O relógio logístico começa a correr, mas o produto fica parado.
* **Impacto de Categorias:** Itens de "Móveis e Decoração" possuem alto índice de sinistro logístico devido à complexidade de cubagem e peso, exigindo transportadoras especializadas.
* **Cancelamento vs. Atraso:** A taxa de cancelamento dispara exponencialmente quando o pedido supera 5 dias de atraso.

### 3. Machine Learning 🤖
* **Objetivo:** Regressão para prever `dias_de_atraso` (ou margem de segurança).
* **Algoritmos Testados:** Linear Regression, XGBoost e Random Forest.
* **Modelo Campeão:** `RandomForestRegressor`.
* **Performance:** O modelo alcançou um MAE (Erro Médio Absoluto) competitivo, capaz de diferenciar com precisão rotas de risco (ex: SP -> AM) de rotas seguras (ex: SP -> SP).

### 4. Deploy (Aplicação Final)
Desenvolvimento de uma Web App em **Streamlit** simulando uma ferramenta de gestão:
* **Simulador:** O usuário insere origem, destino e dimensões; o modelo retorna a previsão de dias em tempo real.
* **Arquitetura:** Uso de `utils.py` para modularização e garantia de consistência entre o treinamento e a aplicação (Training-Serving Skew prevention).

---

## 🚀 Como Executar o Projeto

### Pré-requisitos
* Python 3.9 ou superior.
* Conta no Kaggle (para download dos dados).

### Passo 1: Instalação
Clone o repositório e instale as dependências:
```bash
git clone [https://github.com/anapaulads/anapaulads-Analise-e-Modelagem-Preditiva-de-Performance-Logistica.git](https://github.com/anapaulads/anapaulads-Analise-e-Modelagem-Preditiva-de-Performance-Logistica.git)
cd olist-logistics
pip install -r requirements.txt
```

### Passo 2: Configuração da API Kaggle (Dados)
Este projeto baixa os dados brutos automaticamente. Para isso, você precisa da chave de API:
Para que o download automático dos dados funcione:
1. Crie uma conta no Kaggle.
2. Vá em 'Settings' > 'API' > 'Create New Token'.
3. Um arquivo `kaggle.json` será baixado.
4. Coloque esse arquivo na pasta raiz deste projeto. 

### Passo 3: Executando
Para abrir o Dashboard no seu navegador:
```bash
streamlit run app.py
```

## 🗂 Estrutura de Arquivos
```text
├── data/                  # Armazena os CSVs (Ignorado no Git, baixado via script)
├── models/                # Modelo treinado (.pkl) (Ignorado no Git)
├── notebooks/             # Jupyter Notebooks de desenvolvimento
│   ├── ETL_EDA_Logistics_Analytics.ipynb
│   └── Modelagem_Logistica.ipynb
├── utils/                 # Funções compartilhadas (ETL e App)
│   └── utils.py
├── app.py                 # Aplicação Streamlit (Dashboard + Simulador)
├── kaggle.json            # Credenciais do Kaggle (Adicione o seu aqui)
├── requirements.txt       # Bibliotecas necessárias para rodar o projeto
└── README.md              # Documentação do projeto
```

## Contribuições
Sugestões, melhorias e novas ideias são bem-vindas!  
Sinta-se à vontade para abrir issues ou pull requests.

## ✒️ Autor

**Ana Paula Dias** *Data Scientist | Data Analyst*

Entre em contacto para discutir este projeto ou oportunidades:

[![LinkedIn](https://img.shields.io/badge/-LinkedIn-blue?style=flat-square&logo=Linkedin&logoColor=white)]([SEU_URL_DO_LINKEDIN_AQUI](https://www.linkedin.com/in/anapauladss/))
[![Gmail](https://img.shields.io/badge/-Gmail-c14438?style=flat-square&logo=Gmail&logoColor=white)](mailto:contato.paulla@outlook.com)

---

## 📄 Licença
Este projeto está licenciado sob a Licença MIT - consulte o arquivo [LICENSE](LICENSE) para mais detalhes.

## 🙏 Agradecimentos
* **Olist:** Pela disponibilização pública do [Brazilian E-Commerce Public Dataset](https://www.kaggle.com/olistbr/brazilian-ecommerce) no Kaggle, que tornou este estudo possível.