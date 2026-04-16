import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import time

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Bot 5: Logit Fundamental", layout="wide", page_icon="🏛️")

# --- ESTILOS PERSONALIZADOS ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- REGLAS DE ESTRATEGIA (Sidebar) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2534/2534351.png", width=80)
    st.header("🏛️ Estrategia Bot 5")
    st.info("""
    **Horizonte:** 1 - 3 Meses  
    **Stop Loss:** 12%  
    **Take Profit:** 25%  
    **Modelo:** Arbitraje Logit de Panel Data.
    """)
    
    st.divider()
    st.subheader("Configuración de Vista")
    
    # Lista Maestra Unificada
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

# --- MOTOR DE CÁLCULO (Caching para evitar lentitud) ---
@st.cache_data(ttl=43200) # Se actualiza cada 12 horas
def get_logit_data(ticker_list):
    raw_data = []
    progress_bar = st.progress(0)
    for i, ticker in enumerate(ticker_list):
        try:
            t = yf.Ticker(ticker)
            info = t.info
            if not info or 'sector' not in info: continue
            raw_data.append({
                "Ticker": ticker,
                "Sector": info.get('sector', 'Other'),
                "MarketCap": info.get('marketCap', np.nan),
                "PE_Fwd": info.get('forwardPE', np.nan),
                "Growth": info.get('revenueGrowth', np.nan),
                "ROE": info.get('returnOnEquity', np.nan),
                "Margin": info.get('profitMargins', np.nan),
                "D_E": info.get('debtToEquity', 100) / 100,
                "Ebitda_G": info.get('earningsGrowth', np.nan)
            })
            progress_bar.progress((i + 1) / len(ticker_list))
        except: continue
    
    df = pd.DataFrame(raw_data)
    for col in ['MarketCap', 'PE_Fwd', 'Growth', 'ROE', 'Margin', 'Ebitda_G']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['PE_Fwd', 'Growth', 'ROE'])
    df['Log_MC'] = np.log(df['MarketCap'].replace(0, np.nan))

    cols_z = ['PE_Fwd', 'Growth', 'ROE', 'Margin', 'Ebitda_G']
    for col in cols_z:
        df[f'z_{col}'] = df.groupby('Sector')[col].transform(
            lambda x: (x - x.mean()) / x.std() if len(x) > 1 and x.std() > 0 else 0
        )

    df['logit_z'] = (0.5 + (df['z_Growth'].fillna(0) * 1.5) + (df['z_Ebitda_G'].fillna(0) * 1.0) +
                     (df['z_ROE'].fillna(0) * 1.2) + (df['z_Margin'].fillna(0) * 0.8) +
                     (df['Log_MC'].fillna(df['Log_MC'].mean()) * 0.25) - 
                     (df['z_PE_Fwd'].fillna(0) * 1.8) - (df['D_E'].fillna(1) * 0.7))
    
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

# --- UI PRINCIPAL ---
st.title("🏛️ Radar de Valor Logit")
st.markdown("---")

if st.button("🚀 Ejecutar Escaneo Semanal"):
    df_final = get_logit_data(TICKERS)
    
    # KPIs Rápidos
    c1, c2, c3 = st.columns(3)
    gangas = df_final[df_final['Estatus'].str.contains("7")]
    c1.metric("Gangas Encontradas", len(gangas))
    c2.metric("Top Ticker", gangas.iloc[0]['Ticker'] if not gangas.empty else "N/A")
    c3.metric("Confianza Media", f"{df_final['Prob_Ganga'].mean()*100:.1f}%")

    st.divider()

    # Tabla Filtrada
    df_filtered = df_final[df_final['Estatus'].isin(selected_status)]
    st.subheader(f"🎯 Selección por Convicción ({len(df_filtered)} activos)")
    
    st.dataframe(
        df_filtered[['Ticker', 'Sector', 'Estatus', 'Prob_Ganga', 'PE_Fwd', 'Growth', 'ROE']],
        column_config={
            "Prob_Ganga": st.column_config.ProgressColumn("Probabilidad", format="%.2f", min_value=0, max_value=1),
            "PE_Fwd": "P/E Proyectado",
            "Growth": st.column_config.NumberColumn("Crecimiento", format="%.2%"),
            "ROE": st.column_config.NumberColumn("ROE", format="%.2%")
        },
        use_container_width=True,
        hide_index=True
    )

    # Gráfico de Dispersión
    st.divider()
    st.subheader("📈 Mapa de Oportunidades: Valuación vs Probabilidad")
    fig = px.scatter(df_filtered, x="PE_Fwd", y="Prob_Ganga", color="Sector", size="ROE",
                     hover_name="Ticker", title="Buscando el cuadrante inferior-derecho (Barato + Alta Prob)")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Haz clic en el botón para iniciar el análisis econométrico.")
