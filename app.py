import pandas as pd
import streamlit as st
from collections import Counter
from typing import List, Optional, Tuple, Any

# --- Configuración de la Página de Streamlit ---
# Usamos st.set_page_config() para establecer el título de la pestaña del navegador, el ícono y el layout.
# Debe ser el primer comando de Streamlit en tu script.
st.set_page_config(
    page_title="🔮 Predicción de Lotería",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Funciones de Lógica de la Aplicación ---

def safe_str_split_to_int(series: pd.Series) -> List[List[int]]:
    """
    Convierte una serie de strings con números separados por espacios a una lista de listas de enteros.
    Maneja de forma segura los valores no válidos o vacíos.
    """
    processed_list = []
    for item in series:
        if isinstance(item, str):
            # Filtra solo los elementos que son dígitos antes de convertir a int
            numbers = [int(n) for n in item.split() if n.isdigit()]
            processed_list.append(numbers)
        else:
            # Si el valor no es un string (ej. NaN o ya es un número), añade una lista vacía.
            processed_list.append([])
    return processed_list

def safe_to_int(series: pd.Series) -> pd.Series:
    """
    Convierte una serie a números enteros, manejando errores.
    Los valores no numéricos se convierten en None (NaN para pandas).
    """
    return pd.to_numeric(series, errors='coerce').astype('Int64')

@st.cache_data # Decorador de Streamlit para optimizar el rendimiento.
def load_and_process_data(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Carga y procesa el archivo Excel subido por el usuario.
    La función está cacheada: solo se re-ejecutará si el archivo subido cambia.
    
    Retorna un DataFrame procesado o None si ocurre un error.
    """
    try:
        df = pd.read_excel(uploaded_file)
        
        # --- Validación de Columnas ---
        required_columns = ['Bolillas', 'YaPa', 'Adicionales']
        if not all(col in df.columns for col in required_columns):
            st.error(f"El archivo Excel debe contener las columnas: {', '.join(required_columns)}")
            return None

        # --- Procesamiento Robusto de Datos ---
        df_processed = df.copy()
        df_processed['Bolillas'] = safe_str_split_to_int(df_processed['Bolillas'].astype(str))
        df_processed['YaPa'] = safe_to_int(df_processed['YaPa'])
        df_processed['Adicionales'] = safe_str_split_to_int(df_processed['Adicionales'].astype(str))
        
        return df_processed

    except Exception as e:
        st.error(f"Ocurrió un error al leer o procesar el archivo: {e}")
        return None

def get_predictions(df: pd.DataFrame, num_bolillas: int, num_yapa: int, num_adicionales: int) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Analiza las frecuencias y devuelve las predicciones basadas en los números más comunes.
    """
    # Aplanar las listas de números en una sola lista para cada categoría
    todas_bolillas = [num for sublist in df['Bolillas'] for num in sublist]
    # Usamos .dropna() para eliminar valores nulos antes de contar
    todas_yapa = [num for num in df['YaPa'].dropna()]
    todas_adicionales = [num for sublist in df['Adicionales'] for num in sublist]

    # Contar la frecuencia de cada número
    freq_bolillas = Counter(todas_bolillas)
    freq_yapa = Counter(todas_yapa)
    freq_adicionales = Counter(todas_adicionales)

    # Obtener los N números más comunes
    pred_bolillas = [num for num, _ in freq_bolillas.most_common(num_bolillas)]
    pred_yapa = [num for num, _ in freq_yapa.most_common(num_yapa)]
    pred_adicionales = [num for num, _ in freq_adicionales.most_common(num_adicionales)]

    return pred_bolillas, pred_yapa, pred_adicionales, freq_bolillas, freq_yapa, freq_adicionales

# --- Interfaz de Usuario (UI) ---

st.title("🔮 Predictor de Lotería Basado en Frecuencia")
st.markdown("""
Sube tu archivo Excel con los resultados históricos de los sorteos. 
La aplicación analizará la frecuencia de cada número y te mostrará los más probables para las categorías **Bolillas, YaPa y Adicionales**.
""")

# --- Barra Lateral para Controles ---
with st.sidebar:
    st.header("⚙️ Configuración")
    
    archivo = st.file_uploader("Sube tu archivo Excel (.xlsx)", type=["xlsx"])
    
    st.subheader("Cantidad de Números a Predecir")
    num_bolillas_pred = st.slider("Bolillas", min_value=1, max_value=20, value=6, step=1)
    num_yapa_pred = st.slider("YaPa", min_value=1, max_value=10, value=1, step=1)
    num_adicionales_pred = st.slider("Adicionales", min_value=1, max_value=10, value=2, step=1)

# --- Lógica Principal de la App ---
if archivo:
    df = load_and_process_data(archivo)
    
    if df is not None:
        st.success("¡Archivo cargado y procesado con éxito!")
        
        pred_bolillas, pred_yapa, pred_adicionales, freq_bolillas, freq_yapa, freq_adicionales = get_predictions(
            df, num_bolillas_pred, num_yapa_pred, num_adicionales_pred
        )
        
        # --- Mostrar Resultados de Predicción ---
        st.header("📊 Tu Predicción Personalizada")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Bolillas", " ".join(map(str, sorted(pred_bolillas))))
        with col2:
            st.metric("YaPa", " ".join(map(str, sorted(pred_yapa))))
        with col3:
            st.metric("Adicionales", " ".join(map(str, sorted(pred_adicionales))))
            
        st.info("Nota: Las predicciones se basan en la frecuencia histórica y no garantizan resultados futuros.", icon="ℹ️")

        # --- Frecuencias Detalladas (en un expander para no saturar la UI) ---
        with st.expander("📈 Ver Análisis de Frecuencia Detallado"):
            st.subheader("Números más frecuentes")
            
            p1, p2, p3 = st.columns(3)
            with p1:
                st.write("**Bolillas:**")
                st.dataframe(pd.DataFrame(freq_bolillas.most_common(15), columns=['Número', 'Frecuencia']))
            with p2:
                st.write("**YaPa:**")
                st.dataframe(pd.DataFrame(freq_yapa.most_common(10), columns=['Número', 'Frecuencia']))
            with p3:
                st.write("**Adicionales:**")
                st.dataframe(pd.DataFrame(freq_adicionales.most_common(10), columns=['Número', 'Frecuencia']))
            
            st.subheader("Vista Previa de los Datos Procesados")
            st.dataframe(df.head())
else:
    st.info("👈 Sube un archivo en la barra lateral para comenzar.")
