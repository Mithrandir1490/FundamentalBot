import streamlit as st
import pandas as pd
import numpy as np

# ==========================================
# CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(page_title="Radar Logit v4.0", layout="wide", page_icon="🏛️")

# Estilo personalizado para las etiquetas de estatus
def color_estatus(val):
    if "7) GANGA" in val: return 'background-color: #004d00; color: white; font-weight: bold'
    if "6) MUY BARATA" in val: return 'background-color: #008000; color: white'
    if "5) BARATA" in val: return 'background-color: #99ff99; color: black'
    if "4) REGULAR" in val: return 'background-color: #ffffcc; color: black'
    if "3) CARA" in val: return 'background-color: #ffcc99; color: black'
    if "2) MUY CARA" in val: return 'background-color: #ff6666; color: white'
    if "1) EVITAR" in val: return 'background-color: #990000; color: white'
    return ''

# ==========================================
# CARGA DE DATOS (LECTURA DE LA LIBRETA)
# ==========================================
@st.cache_data(ttl=3600)
def load_and_process_data():
    try:
        # Leemos el archivo que generó el robot
        df = pd.read_csv("logit_data.csv")
        
        # Limpieza de datos numéricos
        cols_num = ['MarketCap', 'PE_Fwd', 'Growth', 'ROE', 'Margin', 'D_E', 'Ebitda_G']
        for col in cols_num:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Eliminamos filas que no tengan datos fundamentales básicos
        df = df.dropna(subset=['PE_Fwd', 'Growth', 'ROE'], how='all')
        
        # --- PROCESAMIENTO MATEMÁTICO (MODELO LOGIT) ---
        
        # 1. Tamaño de Empresa (Logaritmo del Market Cap)
        df['Log_MC'] = np.log(df['MarketCap'].replace(0, np.nan))
        
        # 2. Z-Scores por Sector (Normalización relativa)
        for col in ['PE_Fwd', 'Growth', 'ROE', 'Margin', 'Ebitda_G']:
            df[f'z_{col}'] = df.groupby('Sector')[col].transform(
                lambda x: (x - x.mean()) / x.std() if len(x) > 1 and x.std() > 0 else 0
            )

        # 3. FÓRMULA MAESTRA LOGIT (Tus pesos de convicción)
        df['logit_z'] = (0.5 + 
                         (df.get('z_Growth', 0) * 1.5) + 
                         (df.get('z_Ebitda_G', 0) * 1.0) +
                         (df.get('z_ROE', 0) * 1.2) + 
                         (df.get('z_Margin', 0) * 0.8) +
                         (df['Log_MC'].fillna(df['Log_MC'].mean()) * 0.25) - 
                         (df.get('z_PE_Fwd', 0) * 1.8) - 
                         (df.get('D_E', 100)/100 * 0.7))
        
        # 4. Cálculo de Probabilidad (Función Sigmoide)
        df['Prob_Ganga'] = 1 / (1 + np.exp(-df['logit_z'].astype(float)))
        
        # 5. Clasificación por Estatus
        def semaforo(p):
            if p >= 0.90: return "💎 7) GANGA"
            if p >= 0.80: return "🔥 6) MUY BARATA"
            if p >= 0.70: return "✅ 5) BARATA"
            if p >= 0.50: return "➖ 4) REGULAR"
            if p >= 0.30: return "⚠️ 3) CARA"
            if p >= 0.15: return "🚨 2) MUY CARA"
            return "❌ 1) EVITAR"

        df['Estatus'] = df['Prob_Ganga'].apply(semaforo)
        return df.sort_values("Prob_Ganga", ascending=False)
    
    except Exception as e:
        st.error(f"Error cargando los datos: {e}")
        return pd.DataFrame()

# ==========================================
# INTERFAZ DE USUARIO (UI)
# ==========================================
st.title("🏛️ Radar de Valor Logit Pro")
st.markdown("### El Ejército de Bots: Inteligencia Fundamental Automatizada")

df_final = load_and_process_data()

if df_final.empty:
    st.warning("⚠️ Esperando la primera recolección de datos del robot. Asegúrate de que 'logit_data.csv' exista en tu repositorio.")
else:
    # Sidebar con métricas de resumen
    with st.sidebar:
        st.header("📊 Resumen del Mercado")
        st.metric("Activos Analizados", len(df_final))
        gangas = len(df_final[df_final['Prob_Ganga'] >= 0.80])
        st.metric("Oportunidades (💎/🔥)", gangas)
        st.divider()
        st.info("Los datos se actualizan automáticamente todos los lunes a las 9:00 AM CST.")

    # Filtros rápidos
    col1, col2 = st.columns([2, 1])
    with col1:
        sectores = ["Todos"] + sorted(df_final['Sector'].unique().tolist())
        sector_sel = st.selectbox("Filtrar por Sector:", sectores)
    
    with col2:
        estatus_filtro = st.multiselect("Filtrar por Estatus:", 
                                        df_final['Estatus'].unique().tolist(),
                                        default=None)

    # Aplicar filtros
    df_display = df_final.copy()
    if sector_sel != "Todos":
        df_display = df_display[df_display['Sector'] == sector_sel]
    if estatus_filtro:
        df_display = df_display[df_display['Estatus'].isin(estatus_filtro)]

    # Mostrar Tabla Maestra
    st.subheader(f"Resultados: {len(df_display)} activos")
    
    # Formateo de la tabla
    st.dataframe(
        df_display[['Ticker', 'Sector', 'Estatus', 'Prob_Ganga', 'PE_Fwd', 'Growth', 'ROE', 'Margin']]
        .style.applymap(color_estatus, subset=['Estatus'])
        .format({
            "Prob_Ganga": "{:.2%}",
            "Growth": "{:.1%}",
            "ROE": "{:.1%}",
            "Margin": "{:.1%}",
            "PE_Fwd": "{:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # --- GLOSARIO ---
    with st.expander("ℹ️ ¿Cómo leer este Radar?"):
        st.write("""
        - **Logit Scoring:** Un modelo econométrico que pondera crecimiento, rentabilidad y múltiplos de valuación frente a su propio sector.
        - **Prob_Ganga:** La probabilidad estadística de que el activo esté subvaluado (Margen de Seguridad).
        - **Estatus:** Clasificación visual basada en la probabilidad acumulada.
        """)

st.markdown("---")
st.caption("Bot Logit v4.0 - Generación de Alpha mediante Arbitraje Estadístico Fundamental.")
