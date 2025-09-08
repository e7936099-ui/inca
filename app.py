import pandas as pd
import streamlit as st
import plotly.express as px
from collections import Counter, defaultdict
from typing import List, Optional, Any, Dict
from itertools import combinations

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
    
    # --- Frecuencia (Números Calientes) ---
    all_bolillas = [num for sublist in df['Bolillas'] for num in sublist]
    freq_bolillas = Counter(all_bolillas)
    freq_yapa = Counter(all_yapa if (all_yapa := [num for num in df['YaPa'].dropna()]) else [])
    freq_adicionales = Counter(all_adicionales if (all_adicionales := [num for sublist in df['Adicionales'] for num in sublist]) else [])

    # --- Recencia (Última Vez Vistos) ---
    last_seen_bolillas = {}
    for index, row in df.iterrows():
        for num in row['Bolillas']:
            if num not in last_seen_bolillas:
                last_seen_bolillas[num] = row['Sorteo_ID']
    df_recencia = pd.DataFrame(list(last_seen_bolillas.items()), columns=['Número', 'Sorteos Atrás']).sort_values(by='Sorteos Atrás').reset_index(drop=True)

    # --- ANÁLISIS DE PATRONES ---
    
    # 1. Pares más comunes
    all_pairs = [pair for sublist in df['Bolillas'] for pair in combinations(sorted(sublist), 2)]
    pairs_counter = Counter(all_pairs)

    # 2. Números siguientes
    following_numbers = defaultdict(Counter)
    for i in range(len(df) - 1):
        current_draw_numbers = df.iloc[i]['Bolillas']
        next_draw_numbers = df.iloc[i+1]['Bolillas']
        for num in current_draw_numbers:
            following_numbers[num].update(next_draw_numbers)

    # 3. Ratio Pares/Impares y Sumas
    odd_even_ratios = []
    sums = []
    for sublist in df['Bolillas']:
        if sublist:
            evens = sum(1 for num in sublist if num % 2 == 0)
            odds = len(sublist) - evens
            odd_even_ratios.append(f"{evens} Pares, {odds} Impares")
            sums.append(sum(sublist))
    
    return {
        "freq_bolillas": freq_bolillas,
        "freq_yapa": freq_yapa,
        "freq_adicionales": freq_adicionales,
        "recencia": df_recencia,
        "pairs": pairs_counter,
        "following_numbers": following_numbers,
        "odd_even_ratios": Counter(odd_even_ratios),
        "sums": sums
    }

# --- Interfaz de Usuario (UI) ---

st.title("🔮 Predictor de Lotería Basado en Estadísticas y Patrones")
st.markdown("""
Analiza el historial de sorteos para descubrir no solo los números más frecuentes, sino también **patrones ocultos**, **pares comunes** y **tendencias** que podrían ayudarte en tu próxima jugada.
""")

with st.sidebar:
    st.header("⚙️ Configuración")
    source_choice = st.radio("Elige la fuente de datos", ("Usar archivo por defecto (inca.xlsx)", "Subir un archivo personalizado"))
    archivo = 'inca.xlsx' if source_choice == "Usar archivo por defecto (inca.xlsx)" else st.file_uploader("Sube tu archivo Excel (.xlsx)", type=["xlsx"])
    
    st.subheader("Cantidad de Números a Predecir")
    num_bolillas_pred = st.slider("Bolillas", 1, 20, 6)
    num_yapa_pred = st.slider("YaPa", 1, 10, 1)
    num_adicionales_pred = st.slider("Adicionales", 1, 10, 2)

if archivo:
    df = load_and_process_data(archivo)
    if df is not None:
        st.success(f"¡Archivo cargado y procesado! Se han analizado **{len(df)}** sorteos.")
        analysis = get_full_analysis(df)
        pred_bolillas = [num for num, _ in analysis['freq_bolillas'].most_common(num_bolillas_pred)]
        pred_yapa = [num for num, _ in analysis['freq_yapa'].most_common(num_yapa_pred)]
        pred_adicionales = [num for num, _ in analysis['freq_adicionales'].most_common(num_adicionales_pred)]
        
        st.header("📊 Tu Predicción Personalizada")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Bolillas", " - ".join(map(str, sorted(pred_bolillas))))
        with col2: st.metric("YaPa", " - ".join(map(str, sorted(pred_yapa))))
        with col3: st.metric("Adicionales", " - ".join(map(str, sorted(pred_adicionales))))
            
        with st.container(border=True):
            st.subheader("💡 ¿Por qué estos números?")
            st.markdown(f"""
            Esta predicción se basa en los **"números calientes"** (los que más han aparecido). Por ejemplo, el **{pred_bolillas[0]}** ha salido **{analysis['freq_bolillas'][pred_bolillas[0]]}** veces.
            Para un análisis más profundo, explora la pestaña **"Análisis de Patrones"**.
            
            *Recuerda que la lotería es un juego de azar. Este análisis es una herramienta estadística y no garantiza resultados futuros.*
            """)

        st.header("🔍 Análisis Profundo de los Datos")
        tab1, tab2, tab3, tab4 = st.tabs(["🔥 Calientes", "❄️ Fríos", "⏰ Recencia", "🔎 Patrones"])

        with tab1:
            # ... (contenido sin cambios)
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
            # ... (contenido sin cambios)
            st.subheader("Los Números que Menos Han Salido")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.write("**Bolillas:**"); st.dataframe(pd.DataFrame(analysis['freq_bolillas'].most_common()[:-16:-1], columns=['Número', 'Frecuencia']))
            with c2:
                st.write("**YaPa:**"); st.dataframe(pd.DataFrame(analysis['freq_yapa'].most_common()[:-11:-1], columns=['Número', 'Frecuencia']))
            with c3:
                st.write("**Adicionales:**"); st.dataframe(pd.DataFrame(analysis['freq_adicionales'].most_common()[:-11:-1], columns=['Número', 'Frecuencia']))

        with tab3:
            st.subheader("Análisis de Recencia de Bolillas")
            st.markdown("Cuántos sorteos han pasado desde la última vez que apareció cada bolilla. '0' = último sorteo.")
            st.dataframe(analysis['recencia'], use_container_width=True)

        with tab4:
            st.subheader("Pares de Bolillas Más Comunes")
            st.markdown("Estos son los dúos de números que más veces han aparecido juntos en el mismo sorteo.")
            df_pairs = pd.DataFrame(analysis['pairs'].most_common(15), columns=['Par', 'Frecuencia'])
            df_pairs['Par'] = df_pairs['Par'].astype(str)
            st.dataframe(df_pairs, use_container_width=True)

            st.subheader("Probabilidad de Números Siguientes")
            st.markdown("Selecciona un número para ver cuáles son los que más probablemente salgan en el **sorteo siguiente**.")
            sorted_bolillas = sorted(analysis['freq_bolillas'].keys())
            selected_num = st.selectbox("Elige un número para analizar:", options=sorted_bolillas, index=sorted_bolillas.index(pred_bolillas[0]) if pred_bolillas[0] in sorted_bolillas else 0)
            if selected_num in analysis['following_numbers']:
                following_data = analysis['following_numbers'][selected_num]
                df_following = pd.DataFrame(following_data.most_common(10), columns=['Número Siguiente', 'Frecuencia'])
                st.dataframe(df_following, use_container_width=True)
            else:
                st.warning(f"El número {selected_num} no tiene datos de sorteos siguientes (podría ser del último sorteo registrado).")
            
            st.subheader("Estadísticas Generales del Sorteo")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Ratio Pares / Impares Más Común**")
                df_ratios = pd.DataFrame(analysis['odd_even_ratios'].most_common(5), columns=['Combinación', 'Frecuencia'])
                st.dataframe(df_ratios)
            with col2:
                st.markdown("**Distribución de la Suma de Bolillas**")
                if analysis['sums']:
                    fig_sums = px.histogram(pd.DataFrame(analysis['sums'], columns=['Suma']), x='Suma', nbins=30, title="Frecuencia de la Suma Total de Bolillas")
                    st.plotly_chart(fig_sums, use_container_width=True)
                else:
                    st.write("No hay datos de suma para mostrar.")

        with st.expander("Vista Previa de los Datos Procesados"):
            st.dataframe(df.head())
else:
    st.info("👈 Elige una fuente de datos en la barra lateral para comenzar el análisis.")

