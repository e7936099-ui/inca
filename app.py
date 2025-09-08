import pandas as pd
import streamlit as st
import plotly.express as px
from collections import Counter
from typing import List, Optional, Tuple, Any, Dict

# --- Configuración de la Página de Streamlit ---
st.set_page_config(
    page_title="🔮 Predicción de Lotería",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- Funciones de Lógica de la Aplicación ---

def safe_str_split_to_int(series: pd.Series) -> List[List[int]]:
    """Convierte una serie de strings a una lista de listas de enteros de forma segura."""
    processed_list = []
    for item in series:
        if isinstance(item, str):
            numbers = [int(n) for n in item.split() if n.isdigit()]
            processed_list.append(numbers)
        else:
            processed_list.append([])
    return processed_list

def safe_to_int(series: pd.Series) -> pd.Series:
    """Convierte una serie a enteros, manejando errores."""
    return pd.to_numeric(series, errors='coerce').astype('Int64')

@st.cache_data
def load_and_process_data(file_source) -> Optional[pd.DataFrame]:
    """
    Carga y procesa el archivo Excel desde una ruta o un objeto subido.
    Cacheado para optimizar rendimiento.
    """
    try:
        df = pd.read_excel(file_source)
        required_columns = ['Bolillas', 'YaPa', 'Adicionales']
        if not all(col in df.columns for col in required_columns):
            st.error(f"El archivo Excel debe contener las columnas: {', '.join(required_columns)}")
            return None

        df_processed = df.copy()
        # Añadir un índice de sorteo para el análisis de recencia
        df_processed['Sorteo_ID'] = range(len(df_processed) - 1, -1, -1)
        
        df_processed['Bolillas'] = safe_str_split_to_int(df_processed['Bolillas'].astype(str))
        df_processed['YaPa'] = safe_to_int(df_processed['YaPa'])
        df_processed['Adicionales'] = safe_str_split_to_int(df_processed['Adicionales'].astype(str))
        
        return df_processed
    except FileNotFoundError:
        st.error("No se pudo encontrar el archivo 'inca.xlsx' en el repositorio. Por favor, sube un archivo personalizado.")
        return None
    except Exception as e:
        st.error(f"Ocurrió un error al leer o procesar el archivo: {e}")
        return None

def get_full_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Realiza un análisis completo de frecuencia, recencia y números fríos."""
    
    # --- Frecuencia (Números Calientes) ---
    all_bolillas = [num for sublist in df['Bolillas'] for num in sublist]
    all_yapa = [num for num in df['YaPa'].dropna()]
    all_adicionales = [num for sublist in df['Adicionales'] for num in sublist]

    freq_bolillas = Counter(all_bolillas)
    freq_yapa = Counter(all_yapa)
    freq_adicionales = Counter(all_adicionales)

    # --- Recencia (Última Vez Vistos) ---
    last_seen_bolillas = {}
    for index, row in df.iterrows():
        for num in row['Bolillas']:
            if num not in last_seen_bolillas:
                last_seen_bolillas[num] = row['Sorteo_ID']
    
    # Convertir a DataFrame para mejor visualización
    df_recencia = pd.DataFrame(list(last_seen_bolillas.items()), columns=['Número', 'Sorteos Atrás'])
    df_recencia = df_recencia.sort_values(by='Sorteos Atrás').reset_index(drop=True)

    return {
        "freq_bolillas": freq_bolillas,
        "freq_yapa": freq_yapa,
        "freq_adicionales": freq_adicionales,
        "recencia": df_recencia
    }

# --- Interfaz de Usuario (UI) ---

st.title("🔮 Predictor de Lotería Basado en Estadísticas")
st.markdown("""
Sube tu archivo Excel con el historial de sorteos para obtener una predicción y un análisis detallado. 
Esta herramienta se basa en la frecuencia histórica de los números.
""")

# --- Barra Lateral para Controles ---
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # --- NUEVO: Selector de fuente de datos ---
    source_choice = st.radio(
        "Elige la fuente de datos",
        ("Usar archivo por defecto (inca.xlsx)", "Subir un archivo personalizado")
    )

    archivo = None
    if source_choice == "Usar archivo por defecto (inca.xlsx)":
        archivo = 'inca.xlsx'
    else:
        archivo = st.file_uploader("Sube tu archivo Excel (.xlsx)", type=["xlsx"])
    
    st.subheader("Cantidad de Números a Predecir")
    num_bolillas_pred = st.slider("Bolillas", min_value=1, max_value=20, value=6, step=1)
    num_yapa_pred = st.slider("YaPa", min_value=1, max_value=10, value=1, step=1)
    num_adicionales_pred = st.slider("Adicionales", min_value=1, max_value=10, value=2, step=1)

# --- Lógica Principal de la App ---
if archivo:
    df = load_and_process_data(archivo)
    
    if df is not None:
        st.success(f"¡Archivo cargado y procesado! Se han analizado **{len(df)}** sorteos.")
        
        analysis = get_full_analysis(df)
        
        # Obtener predicciones
        pred_bolillas = [num for num, _ in analysis['freq_bolillas'].most_common(num_bolillas_pred)]
        pred_yapa = [num for num, _ in analysis['freq_yapa'].most_common(num_yapa_pred)]
        pred_adicionales = [num for num, _ in analysis['freq_adicionales'].most_common(num_adicionales_pred)]
        
        # --- Mostrar Resultados de Predicción ---
        st.header("📊 Tu Predicción Personalizada")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Bolillas", " - ".join(map(str, sorted(pred_bolillas))))
        with col2:
            st.metric("YaPa", " - ".join(map(str, sorted(pred_yapa))))
        with col3:
            st.metric("Adicionales", " - ".join(map(str, sorted(pred_adicionales))))
            
        with st.container(border=True):
            st.subheader("💡 ¿Por qué estos números?")
            explanation = f"""
            Esta predicción se basa en los números que más han aparecido en el historial de sorteos que subiste. Son considerados **"números calientes"**.

            - **Para Bolillas**, los números `{', '.join(map(str, sorted(pred_bolillas)))}` son los más frecuentes. Por ejemplo, el número **{pred_bolillas[0]}** ha salido **{analysis['freq_bolillas'][pred_bolillas[0]]}** veces.
            - **Para YaPa**, el **{pred_yapa[0]}** es el rey, con **{analysis['freq_yapa'][pred_yapa[0]]}** apariciones.
            
            Recuerda que esto es un análisis estadístico y no garantiza resultados futuros. ¡Mucha suerte!
            """
            st.markdown(explanation)

        # --- Análisis Profundo con Pestañas ---
        st.header("🔍 Análisis Profundo de los Datos")
        tab1, tab2, tab3 = st.tabs(["🔥 Números Calientes (Más Frecuentes)", "❄️ Números Fríos (Menos Frecuentes)", "⏰ Última Vez Vistos"])

        with tab1:
            st.subheader("Top de Números por Frecuencia")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.write("**Bolillas:**")
                st.dataframe(pd.DataFrame(analysis['freq_bolillas'].most_common(15), columns=['Número', 'Frecuencia']))
            with c2:
                st.write("**YaPa:**")
                st.dataframe(pd.DataFrame(analysis['freq_yapa'].most_common(10), columns=['Número', 'Frecuencia']))
            with c3:
                st.write("**Adicionales:**")
                st.dataframe(pd.DataFrame(analysis['freq_adicionales'].most_common(10), columns=['Número', 'Frecuencia']))
            
            # Gráfico de Frecuencias
            df_freq_bolillas = pd.DataFrame(analysis['freq_bolillas'].most_common(20), columns=['Número', 'Frecuencia']).sort_values(by='Frecuencia')
            df_freq_bolillas['Número'] = df_freq_bolillas['Número'].astype(str)
            fig = px.bar(df_freq_bolillas, x='Frecuencia', y='Número', orientation='h', title='Frecuencia de las 20 Bolillas más comunes')
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Los Números que Menos Han Salido")
            st.markdown("Estos son los números que han aparecido con menor frecuencia. Algunos estrategas prefieren apostar por ellos esperando que 'ya les toque salir'.")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.write("**Bolillas:**")
                st.dataframe(pd.DataFrame(analysis['freq_bolillas'].most_common()[:-16:-1], columns=['Número', 'Frecuencia']))
            with c2:
                st.write("**YaPa:**")
                st.dataframe(pd.DataFrame(analysis['freq_yapa'].most_common()[:-11:-1], columns=['Número', 'Frecuencia']))
            with c3:
                st.write("**Adicionales:**")
                st.dataframe(pd.DataFrame(analysis['freq_adicionales'].most_common()[:-11:-1], columns=['Número', 'Frecuencia']))
        
        with tab3:
            st.subheader("Análisis de Recencia de Bolillas")
            st.markdown("Esta tabla muestra cuántos sorteos han pasado desde la última vez que apareció cada bolilla. Un valor de '0' significa que salió en el sorteo más reciente.")
            st.dataframe(analysis['recencia'], use_container_width=True)
            
        with st.expander("Vista Previa de los Datos Procesados"):
            st.dataframe(df.head())
else:
    st.info("👈 Elige una fuente de datos en la barra lateral para comenzar el análisis.")
