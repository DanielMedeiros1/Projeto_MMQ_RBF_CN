import streamlit as st
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import io
import os
from scipy.spatial.distance import cdist

# --- 1. Funções de Processamento Adaptadas do Notebook ---

# Constantes definidas no seu notebook
P = 4  # Número de pesos (MMQ)
N_MMQ = 498 # Número de pontos ajustados (y[2:] com y[n-1], y[n-2], u[n], u[n-1])

# Parâmetros de exemplo para RBF (adaptáveis se seu notebook contiver os valores exatos)
NUM_CENTROS = 10 
SIGMA = 1.0 

def process_mmq(data):
    """Executa o ajuste de Curvas via MMQ."""
    
    # 1. Separar as colunas
    try:
        t = data[:, 0]
        u = data[:, 1]
        y = data[:, 2]
    except IndexError:
        st.error("O arquivo DAT não tem as 3 colunas esperadas (t, u, y).")
        return None, None, None, None, None, None
    
    # 2. Construção da matriz X para o MMQ (Regressores)
    # y[n] = w0*y[n-1] + w1*y[n-2] + w2*u[n] + w3*u[n-1]
    
    # y_alvo (Y) começa em y[2]
    Y = y[2:]  
    
    # Matriz de Regressores X
    X = np.column_stack([y[1:-1], y[:-2], u[2:], u[1:-1]])
    
    # 3. Cálculo dos pesos (w_MMQ) e estimativa
    # w_MMQ = pinv(X) @ Y
    w_MMQ = np.dot(la.pinv(X), Y) 
    y_est_MMQ = X @ w_MMQ 
    
    # 4. Cálculo do EMQ
    EMQ_MMQ = np.mean((Y - y_est_MMQ)** 2)
    
    return t, y, Y, y_est_MMQ, EMQ_MMQ, w_MMQ


def process_rbf(data, num_centros=NUM_CENTROS, sigma=SIGMA):
    """Executa o ajuste de Curvas via RBF."""
    
    try:
        y = data[:, 2]
        u = data[:, 1]
    except IndexError:
        return None, None, None, None, None
    
    # 1. Definir Regressores (baseados nos mesmos do MMQ para comparação)
    Y = y[2:] 
    X = np.column_stack([y[1:-1], y[:-2], u[2:], u[1:-1]])
    
    # 2. Selecionar Centros (Usando K-means simplificado: escolher os primeiros pontos)
    # Para simplicidade e reprodutibilidade, usaremos pontos aleatórios dos dados de entrada
    np.random.seed(42)
    indices_centros = np.random.choice(X.shape[0], num_centros, replace=False)
    centros = X[indices_centros, :]

    # 3. Construção da Matriz de Bases Radiais (Phi)
    # cdist calcula a distância euclidiana entre cada regressor (X) e cada centro
    distancias = cdist(X, centros, metric='euclidean')
    
    # Função Gaussiana: phi_i(x) = exp(-(x - c_i)^2 / (2 * sigma^2))
    PHI = np.exp(-(distancias ** 2) / (2 * sigma**2))

    # 4. Cálculo dos pesos (w_RBF)
    # w_RBF = pinv(PHI) @ Y
    w_RBF = np.dot(la.pinv(PHI), Y)
    y_est_RBF = PHI @ w_RBF

    # 5. Cálculo do EMQ
    EMQ_RBF = np.mean((Y - y_est_RBF)** 2)
    
    return Y, y_est_RBF, EMQ_RBF, num_centros, sigma

# --- 2. Interface Principal do Streamlit ---

st.set_page_config(page_title="Projeto U2: Ajuste de Curvas", layout="wide")
st.title("Ajuste de Curvas (MMQ e RBF) para Sinais .dat")

st.markdown("""
Esta aplicação permite o **upload de qualquer arquivo `.dat`** para análise e ajuste de curvas, utilizando os modelos de **Mínimos Quadrados (MMQ)** e **Funções de Base Radial (RBF)**, conforme o projeto da Unidade II.
""")

st.sidebar.header("1. Upload do Arquivo")
uploaded_file = st.sidebar.file_uploader(
    "Escolha o seu arquivo .dat", type=["dat"]
)

if uploaded_file is not None:
    # Ler o arquivo .dat como texto e converter para um array NumPy
    try:
        # Decodificar o conteúdo para string e usar io.StringIO para carregar no numpy
        string_data = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        data = np.loadtxt(string_data)
        
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: Certifique-se de que é um arquivo .dat válido e formatado corretamente (t, u, y). Detalhe: {e}")
        st.stop()

    st.sidebar.success(f"Arquivo '{uploaded_file.name}' carregado com sucesso.")
    st.sidebar.markdown("---")
    
    # --- Configurações RBF ---
    st.sidebar.header("2. Parâmetros RBF")
    num_centros = st.sidebar.slider("Número de Centros (RBF)", 2, 50, NUM_CENTROS)
    sigma = st.sidebar.number_input("Valor de Sigma (RBF)", min_value=0.1, max_value=10.0, value=SIGMA, step=0.1)

    # --- Execução dos Processamentos ---
    
    # MMQ
    t, y_original, Y_alvo, y_est_MMQ, EMQ_MMQ, w_MMQ = process_mmq(data)
    
    # RBF
    Y_alvo_RBF, y_est_RBF, EMQ_RBF, _, _ = process_rbf(data, num_centros, sigma)
    
    # Apenas prossegue se o MMQ retornou valores válidos
    if Y_alvo is not None:
        
        N = len(Y_alvo) # Número de pontos ajustados (498)
        
        st.header("1. Ajuste de Curvas e Resultados")
        
        # --- Gráfico de Comparação MMQ vs RBF ---
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # O eixo de tempo para o ajuste é t[2:]
        t_ajuste = t[2:N+2]

        ax.plot(t_ajuste, Y_alvo, color="blue", label='Dados Reais (y)') 
        ax.plot(t_ajuste, y_est_MMQ, "-.r", label=f'Ajuste MMQ (EMQ: {EMQ_MMQ:.4f})')
        ax.plot(t_ajuste, y_est_RBF, "--g", label=f'Ajuste RBF (EMQ: {EMQ_RBF:.4f})')

        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Saída (Amplitude)")
        ax.set_title(f"Comparação de Ajustes: MMQ vs RBF | Arquivo: {uploaded_file.name}")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)

        # --- Tabela de Resultados ---
        st.subheader("Métricas de Erro")
        
        resultados_data = {
            'Modelo': ['MMQ', 'RBF'],
            'EMQ': [f"{EMQ_MMQ:.8f}", f"{EMQ_RBF:.8f}"],
            'Melhor?': ['SIM' if EMQ_MMQ < EMQ_RBF else 'NÃO', 'SIM' if EMQ_RBF < EMQ_MMQ else 'NÃO']
        }
        
        st.table(resultados_data)
        
        # --- Análise e Conclusões ---
        st.subheader("Análise Detalhada")
        
        st.markdown("**MMQ (Mínimos Quadrados):**")
        st.code(f"w_MMQ = {w_MMQ.tolist()}")
        st.markdown(f"**Equação do MMQ (com os pesos calculados):**")
        st.latex(
            f"\\hat{{y}}[n] = {w_MMQ[0]:.4f} \\cdot y[n-1] + {w_MMQ[1]:.4f} \\cdot y[n-2] + {w_MMQ[2]:.4f} \\cdot u[n] + {w_MMQ[3]:.4f} \\cdot u[n-1]")

        st.markdown("**Conclusão do seu Projeto:**")
        st.markdown("O seu projeto indica que a RBF, por ser não-linear, tem maior capacidade de generalização e robustez a ruídos e *outliers* do que o MMQ linear, sendo mais indicada para sistemas complexos. No entanto, o melhor modelo é o que apresentar o menor EMQ para o sinal carregado.")