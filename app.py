import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Bot 5: Logit Fundamental", layout="wide", page_icon="🏛️")

# --- REGLAS (Sidebar) ---
with st.sidebar:
    st.header("🏛️ Estrategia Bot 5")
    st.info("**Horizonte:** 1-3 Meses | **SL:** 12% | **TP:** 25%")
    
    # Lista Maestra
    TICKERS = list(set([
        "ADYEN.AS", "UBER", "ADP", "DSY.PA", "UNH", "TEM", "OSCR", "HIMS", "DECK", "ADBE", "ACN", "DLO", "FDS", "WKL.AS", "LULU", "NVO", 
        "GEV", "BE", "VRT", "CEG", "NEE", "SRE", "VST", "V", "MA", "MCO", "SPGI", "ISRG", "AXON", "ABNB", "ANET", "BSX", 
        "TTD", "NOW", "CRM", "SCHW", "BLK", "GS", "XOM", "CVX", "CAT", "DE", "FIX", "ETN", "HON", "WM", "SMCI", "ALAB", 
        "CORT", "ONTO", "AX", "VIV", "GHM", "SLAB", "LSCC", "LASR", "SITM", "MCHP", "MRVL", "BAM", "DHR", 
        "QSR", "BABA", "GE", "CPNG", "EXPE", "ROK", "ZBRA", "CGNX", "PATH", "PEGA", "MDT", "PRCT", "OMCL", "SYK", "TER", "LECO", 
        "OII", "FARO", "PTC", "QCOM", "AVAV", "TDY", "KTOS", "NOC", "GD", "RTX", "LHX", "APP", 
        "IREN", "AMAT", "KLAC", "RMBS", "SIMO", "ARM", "SNPS", "CRDO", "GLW", "AMKR", "PWR", "CCJ", "BWXT", 
        "UUUU", "TMQ", "UAMY", "MP", "FCX", "TECK", "SCCO", "IONQ", "RGTI", "COIN", "SPOT", "DDOG", "RXRX", "POET", 
        "RBLX", "CRCL", "BMNR", "ACHR", "BEAM", "MOH", "ENB", "TOST", "AMGN", "FOX", "UTHR", "GOLD", "WBA", 
        "JNJ", "HD", "ABBV", "O", "BLDR", "TPL", "FICO", "DPZ", "URI", "BKNG", "MNST", "WDAY", "SOFI", "NU",
        "NVDA", "AMD", "TSM", "AVGO", "MU", "ASML", "LRCX", "PANW", "CRWD", "FTNT", "ZS", "OKTA", "SNOW", "PLTR", "LLY", "VRTX", "REGN",
        "AAPL", "MSFT", "GOOGL", "META", "AMZN", "MARA", "RIOT", "WMT", "TGT", "COST", "NFLX", "TSLA", "PYPL", "SHOP", "SE"
    ]))

    selected_status = st.multiselect(
        "Filtrar por Estatus:",
        ["💎 7) GANGA", "🔥 6) MUY BARATA", "✅ 5) BARATA", "➖ 4) REGULAR", "⚠️ 3) CARA", "🚨 2) MUY CARA", "❌ 1) EVITAR"],
        default=["💎 7) GANGA", "🔥 6) MUY BARATA", "✅ 5) BARATA"]
    )

# --- MOTOR DE CÁLCULO ---
@st.cache_data(ttl=43200)
def get_logit_data(ticker_list):
    raw_data = []
    progress_bar = st.progress(0)
    
    # Definimos las columnas que esperamos para asegurar que siempre existan
    expected_cols = ['MarketCap', 'PE_Fwd', 'Growth', 'ROE', 'Margin', 'D_E', 'Ebitda_G']
    
    for i, ticker in enumerate(ticker_list):
        try:
            t = yf.Ticker(ticker)
            info = t.info
            if not info or 'sector' not in info: continue
            
            # Extraemos con .get y valor por defecto NaN para evitar KeyErrors posteriores
            raw_data.append({
                "Ticker": ticker,
                "Sector": info.get('sector', 'Other'),
                "MarketCap": info.get('marketCap', np.nan),
                "PE_Fwd": info.get('forwardPE', np.nan),
                "Growth": info.get('revenueGrowth', np.nan),
                "ROE": info.get('returnOnEquity', np.nan),
                "Margin": info.get('profitMargins', np.nan),
                "D_E": info.get('debtToEquity', np.nan),
                "Ebitda_G": info.get('earningsGrowth', np.nan)
            })
            progress_bar.progress((i + 1) / len(ticker_list))
        except: continue
    
    if not raw_data:
        return pd.DataFrame() # Retornar vacío si no hay nada

    df = pd.DataFrame(raw_data)
    
    # Asegurar conversión numérica
    for col in expected_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = np.nan # Si la columna no existe, la creamos vacía

    # Limpieza y cálculos
    df = df.dropna(subset=['PE_Fwd', 'Growth', 'ROE'], how='all')
    if df.empty: return df

    df['Log_MC'] = np.log(df['MarketCap'].replace(0, np.nan))
    
    cols_z = ['PE_Fwd', 'Growth', 'ROE', 'Margin', 'Ebitda_G']
    for col in cols_z:
        if col in df.columns:
            df[f'z_{col}'] = df.groupby('Sector')[col].transform(
                lambda x: (x - x.mean()) / x.std() if len(x) > 1 and x.std() > 0 else 0
            )

    # Función Logit
    df['logit_z'] = (0.5 + (df.get('z_Growth', 0) * 1.5) + (df.get('z_Ebitda_G', 0) * 1.0) +
                     (df.get('z_ROE', 0) * 1.2) + (df.get('z_Margin', 0) * 0.8) +
                     (df['Log_MC'].fillna(df['Log_MC'].mean()) * 0.25) - 
                     (df.get('z_PE_Fwd', 0) * 1.8) - (df.get('D_E', 100)/100 * 0.7))
    
    df['Prob_Ganga'] = 1 / (1 + np.exp(-df['logit_z'].astype(float)))

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

# --- UI ---
st.title("🏛️ Radar de Valor Logit")

if st.button("🚀 Ejecutar Escaneo Semanal"):
    df_final = get_logit_data(TICKERS)
    
    if df_final.empty:
        st.error("No se pudieron obtener datos. Intenta de nuevo en unos minutos (posible bloqueo de API).")
    else:
        # KPIs
        c1, c2, c3 = st.columns(3)
        gangas = df_final[df_final['Estatus'].str.contains("7")]
        c1.metric("Gangas Encontradas", len(gangas))
        c2.metric("Top Ticker", gangas.iloc[0]['Ticker'] if not gangas.empty else "N/A")
        c3.metric("Confianza Media", f"{df_final['Prob_Ganga'].mean()*100:.1f}%")

        # Tabla
        df_filtered = df_final[df_final['Estatus'].isin(selected_status)]
        st.subheader(f"🎯 Selección por Convicción ({len(df_filtered)} activos)")
        st.dataframe(df_filtered[['Ticker', 'Sector', 'Estatus', 'Prob_Ganga', 'PE_Fwd', 'Growth', 'ROE']], use_container_width=True, hide_index=True)
else:
    st.warning("Haz clic en el botón para iniciar.")
