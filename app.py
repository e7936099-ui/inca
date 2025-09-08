import pandas as pd
import streamlit as st
import plotly.express as px
from collections import Counter, defaultdict
from typing import List, Optional, Any, Dict
from itertools import combinations
import numpy as np

# --- Configuración de la Página de Streamlit ---
st.set_page_config(
    page_title="🔮 Predictor de Lotería Estratégico",
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
    """Carga y procesa el archivo Excel."""
    try:
        df = pd.read_excel(file_source)
        required_columns = ['Bolillas', 'YaPa', 'Adicionales']
        if not all(col in df.columns for col in required_columns):
            st.error(f"El archivo Excel debe contener las columnas: {', '.join(required_columns)}")
            return None

        df_processed = df.copy()
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
    """Realiza un análisis completo que incluye frecuencia, recencia y patrones."""
    all_bolillas = [num for sublist in df['Bolillas'] for num in sublist]
    freq_bolillas = Counter(all_bolillas)
    all_numbers = sorted(list(freq_bolillas.keys()))

    last_seen_bolillas = {num: len(df) for num in all_numbers}
    for index, row in df.iterrows():
        for num in row['Bolillas']:
            last_seen_bolillas[num] = row['Sorteo_ID']
    df_recencia = pd.DataFrame(list(last_seen_bolillas.items()), columns=['Número', 'Sorteos Atrás']).sort_values(by='Sorteos Atrás').reset_index(drop=True)

    all_pairs = [pair for sublist in df['Bolillas'] for pair in combinations(sorted(sublist), 2)]
    pairs_counter = Counter(all_pairs)
    
    return {
        "all_numbers": all_numbers,
        "freq_bolillas": freq_bolillas,
        "freq_yapa": Counter([num for num in df['YaPa'].dropna()]),
        "freq_adicionales": Counter([num for sublist in df['Adicionales'] for num in sublist]),
        "recencia": df_recencia,
        "pairs": pairs_counter,
    }

def calculate_recommendations(analysis: Dict[str, Any], w_hot: float, w_cold: float, w_pairs: float) -> pd.DataFrame:
    """Calcula el puntaje de recomendación para cada número."""
    df_scores = pd.DataFrame(analysis['all_numbers'], columns=['Número'])

    # 1. Puntaje Caliente (Frecuencia)
    freq_map = analysis['freq_bolillas']
    df_scores['Frecuencia'] = df_scores['Número'].map(freq_map)
    min_freq, max_freq = df_scores['Frecuencia'].min(), df_scores['Frecuencia'].max()
    df_scores['Puntaje_Caliente'] = (df_scores['Frecuencia'] - min_freq) / (max_freq - min_freq) if max_freq > min_freq else 0

    # 2. Puntaje Frío (Recencia)
    recencia_map = analysis['recencia'].set_index('Número')['Sorteos Atrás']
    df_scores['Sorteos_Atras'] = df_scores['Número'].map(recencia_map)
    min_rec, max_rec = df_scores['Sorteos_Atras'].min(), df_scores['Sorteos_Atras'].max()
    df_scores['Puntaje_Frio'] = (df_scores['Sorteos_Atras'] - min_rec) / (max_rec - min_rec) if max_rec > min_rec else 0
    
    # 3. Puntaje de Pares
    pairs_map = defaultdict(int)
    for pair, freq in analysis['pairs'].items():
        pairs_map[pair[0]] += freq
        pairs_map[pair[1]] += freq
    df_scores['Puntaje_Pares_Raw'] = df_scores['Número'].map(pairs_map)
    min_pair, max_pair = df_scores['Puntaje_Pares_Raw'].min(), df_scores['Puntaje_Pares_Raw'].max()
    df_scores['Puntaje_Pares'] = (df_scores['Puntaje_Pares_Raw'] - min_pair) / (max_pair - min_pair) if max_pair > min_pair else 0

    # 4. Puntaje Total
    df_scores['Puntaje_Total'] = (
        w_hot * df_scores['Puntaje_Caliente'] +
        w_cold * df_scores['Puntaje_Frio'] +
        w_pairs * df_scores['Puntaje_Pares']
    )
    
    return df_scores.sort_values(by='Puntaje_Total', ascending=False).reset_index(drop=True)


# --- Interfaz de Usuario (UI) ---

st.title("🔮 Predictor de Lotería Estratégico")
st.markdown("Esta herramienta analiza datos históricos para ofrecerte una **recomendación estratégica**. Define tu propia estrategia ajustando los pesos de los diferentes factores de análisis.")

with st.sidebar:
    st.header("⚙️ Configuración de Análisis")
    source_choice = st.radio("Elige la fuente de datos", ("Usar archivo por defecto (inca.xlsx)", "Subir un archivo personalizado"))
    archivo = 'inca.xlsx' if source_choice == "Usar archivo por defecto (inca.xlsx)" else st.file_uploader("Sube tu archivo Excel (.xlsx)", type=["xlsx"])
    
    st.header("⚖️ Define tu Estrategia")
    st.markdown("Ajusta la importancia de cada factor en la predicción.")
    weight_hot = st.slider("🔥 Factor Caliente (Frecuencia)", 0.0, 1.0, 0.4, 0.05)
    weight_cold = st.slider("❄️ Factor Frío (Ausencia)", 0.0, 1.0, 0.3, 0.05)
    weight_pairs = st.slider("🤝 Factor de Pares (Compañerismo)", 0.0, 1.0, 0.3, 0.05)
    
    # Normalizar pesos para que sumen 1
    total_weight = weight_hot + weight_cold + weight_pairs
    if total_weight > 0:
        weight_hot /= total_weight
        weight_cold /= total_weight
        weight_pairs /= total_weight
    
if archivo:
    df = load_and_process_data(archivo)
    if df is not None:
        analysis = get_full_analysis(df)
        
        st.success(f"¡Archivo cargado y procesado! Se han analizado **{len(df)}** sorteos.")
        
        df_recommendations = calculate_recommendations(analysis, weight_hot, weight_cold, weight_pairs)
        
        # --- Predicción Principal ---
        st.header("🔮 Recomendación Estratégica")
        pred_bolillas = df_recommendations['Número'].head(6).tolist()
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.metric("Bolillas Recomendadas (Solo 6)", " - ".join(map(str, sorted(pred_bolillas))))
            # Predicciones simples para YaPa y Adicionales
            pred_yapa = analysis['freq_yapa'].most_common(1)[0][0] if analysis['freq_yapa'] else 'N/A'
            st.metric("YaPa (Más Frecuente)", str(pred_yapa))
            pred_adicionales = [num for num, _ in analysis['freq_adicionales'].most_common(2)]
            st.metric("Adicionales (Más Frecuentes)", " - ".join(map(str, sorted(pred_adicionales))))

        with col2:
            st.subheader("Estrategia Aplicada")
            st.markdown(f"**🔥 Caliente:** `{weight_hot:.0%}`")
            st.markdown(f"**❄️ Frío:** `{weight_cold:.0%}`")
            st.markdown(f"**🤝 Pares:** `{weight_pairs:.0%}`")

        with st.container(border=True):
            st.subheader("💡 ¿Cómo se eligieron estos números?")
            st.markdown(f"""
            La recomendación se basa en un **Puntaje de Probabilidad** calculado para cada número. Este puntaje combina tres análisis clave con la importancia que definiste:

            1.  **Factor Caliente (`{weight_hot:.0%}`):** Prioriza los números que han salido con más frecuencia.
            2.  **Factor Frío (`{weight_cold:.0%}`):** Da más puntos a los números que llevan más tiempo sin salir.
            3.  **Factor de Pares (`{weight_pairs:.0%}`):** Beneficia a los números que suelen salir acompañados de otros números frecuentes.
            
            Los **6 números recomendados** son aquellos con el puntaje total más alto. Abajo puedes ver la tabla de clasificación completa.
            """)

        st.subheader("🏆 Tabla de Clasificación de Números")
        st.dataframe(df_recommendations[[
            'Número', 'Puntaje_Total', 'Puntaje_Caliente', 'Puntaje_Frio', 'Puntaje_Pares', 'Frecuencia', 'Sorteos_Atras'
        ]].head(20), use_container_width=True)


        # --- Análisis Profundo ---
        with st.expander("🔍 Análisis Profundo de los Datos"):
            st.header("Análisis Detallado")
            tab1, tab2, tab3 = st.tabs(["🔥 Frecuencia", "❄️ Recencia", "🤝 Pares"])

            with tab1:
                st.subheader("Top de Números por Frecuencia")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.write("**Bolillas:**"); st.dataframe(pd.DataFrame(analysis['freq_bolillas'].most_common(15), columns=['Número', 'Frecuencia']))
                with c2:
                    st.write("**YaPa:**"); st.dataframe(pd.DataFrame(analysis['freq_yapa'].most_common(10), columns=['Número', 'Frecuencia']))
                with c3:
                    st.write("**Adicionales:**"); st.dataframe(pd.DataFrame(analysis['freq_adicionales'].most_common(10), columns=['Número', 'Frecuencia']))
                df_freq = pd.DataFrame(analysis['freq_bolillas'].most_common(20), columns=['Número', 'Frecuencia']).sort_values(by='Frecuencia')
                df_freq['Número'] = df_freq['Número'].astype(str)
                st.plotly_chart(px.bar(df_freq, x='Frecuencia', y='Número', orientation='h', title='Frecuencia de las 20 Bolillas más comunes'), use_container_width=True)

            with tab2:
                st.subheader("Análisis de Recencia de Bolillas")
                st.markdown("Cuántos sorteos han pasado desde la última vez que apareció cada bolilla. '0' = último sorteo.")
                st.dataframe(analysis['recencia'], use_container_width=True)

            with tab3:
                st.subheader("Pares de Bolillas Más Comunes")
                st.markdown("Estos son los dúos de números que más veces han aparecido juntos en el mismo sorteo.")
                df_pairs = pd.DataFrame(analysis['pairs'].most_common(15), columns=['Par', 'Frecuencia'])
                df_pairs['Par'] = df_pairs['Par'].astype(str)
                st.dataframe(df_pairs, use_container_width=True)

else:
    st.info("👈 Elige una fuente de datos y define tu estrategia en la barra lateral para comenzar.")

