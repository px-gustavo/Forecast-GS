# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Forecast SARIMAX — PX (dados locais/repo + upload fallback)", layout="wide")

# =============================================================================
# CONFIG — caminhos RELATIVOS ao repositório (pasta data/)
# =============================================================================
CONFIG = {
    'ARQUIVO_TOTAL': './data/dados_para_teste.csv',
    'ARQUIVO_SEGMENTOS': './data/Entrega diária por cavaleiro.csv',
    'CONFIDENCE_INTERVAL': 0.80,
    'USAR_LOG': True,
    'APLICAR_TRAVA_MAXIMA': True,
    'TRAVA_PERCENTIL': 0.98
}

# =============================================================================
# Funções utilitárias
# =============================================================================
def normalizar_colunas(df: pd.DataFrame):
    df.columns = [c.strip().upper() for c in df.columns]
    return df

def detectar_coluna_data(df: pd.DataFrame):
    for cand in df.columns:
        if any(chave in cand for chave in ("DIA", "DATA")):
            return cand
    return None

def detectar_coluna_valor_total(df: pd.DataFrame):
    for cand in df.columns:
        if any(chave in cand for chave in ("VENDA", "VENDIDOS", "VALOR", "QTDE", "QTD")):
            return cand
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return num_cols[0] if num_cols else None

def limpar_numerico_robusto(s):
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors='coerce').fillna(0)
    s = (
        s.astype(str)
         .str.replace('R$', '', regex=False)
         .str.replace('.', '', regex=False)
         .str.replace(',', '.', regex=False)
    )
    return pd.to_numeric(s, errors='coerce').fillna(0)

def preparar_exogenas_index(idx: pd.DatetimeIndex):
    exog = pd.DataFrame(index=idx)
    exog['IS_WEEKEND'] = (exog.index.weekday >= 5).astype(int)
    exog['IS_MONTH_START'] = (exog.index.day == 1).astype(int)
    exog['IS_HOLIDAY'] = 0
    exog.loc[(exog.index.month == 12) & (exog.index.day == 25), 'IS_HOLIDAY'] = 1  # Natal
    exog.loc[(exog.index.month == 1) & (exog.index.day == 1), 'IS_HOLIDAY'] = 1    # Ano Novo
    exog['IS_EVENT'] = 0
    return exog

def aplicar_features_basicas(df: pd.DataFrame, col_data: str):
    df = df.copy()
    df[col_data] = pd.to_datetime(df[col_data], dayfirst=True, errors='coerce')
    df.dropna(subset=[col_data], inplace=True)
    df.set_index(col_data, inplace=True)
    df.sort_index(inplace=True)

    col_feriado = next((c for c in df.columns if 'FERIADO' in c), None)
    if col_feriado:
        df['IS_HOLIDAY'] = pd.to_numeric(df[col_feriado], errors='coerce').fillna(0)
    else:
        df['IS_HOLIDAY'] = 0
        df.loc[(df.index.month == 12) & (df.index.day == 25), 'IS_HOLIDAY'] = 1
        df.loc[(df.index.month == 1) & (df.index.day == 1), 'IS_HOLIDAY'] = 1

    col_evento = next((c for c in df.columns if 'EVENTO' in c), None)
    if col_evento:
        df['IS_EVENT'] = pd.to_numeric(df[col_evento], errors='coerce').fillna(0)
    else:
        df['IS_EVENT'] = 0

    df['IS_WEEKEND'] = (df.index.weekday >= 5).astype(int)
    df['IS_MONTH_START'] = (df.index.day == 1).astype(int)
    return df

def treinar_e_prever(series_treino, exog_treino, exog_futuro,
                     usar_log=True, aplicar_trava=True, trava_percentil=0.98, nome="SERIE"):
    series_model = np.log1p(series_treino) if usar_log else series_treino
    model = SARIMAX(series_model,
                    exog=exog_treino,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 7),
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    results = model.fit(disp=False, maxiter=200)
    fc_obj = results.get_forecast(steps=len(exog_futuro), exog=exog_futuro)
    fc_mean = fc_obj.predicted_mean
    if usar_log:
        fc_mean = np.expm1(fc_mean)
    fc_mean = fc_mean.clip(lower=0)

    if aplicar_trava:
        teto_hist = series_treino.quantile(trava_percentil)
        teto_maximo = teto_hist * 1.10
        fc_mean = fc_mean.clip(upper=teto_maximo)

    return fc_mean

def carregar_csv_local(caminho: str):
    """Lê um CSV local/relativo tentando ; e , com UTF-8 ou Latin1."""
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo não encontrado: {os.path.abspath(caminho)}")
    erros = []
    for sep in [';', ',']:
        for enc in ['utf-8', 'latin1']:
            try:
                df = pd.read_csv(caminho, sep=sep, encoding=enc)
                return df, sep, enc
            except Exception as e:
                erros.append(f"{os.path.basename(caminho)} (sep={sep}, enc={enc}): {e}")
    raise ValueError("Falha ao ler o CSV.\n" + "\n".join(erros))

def carregar_csv_robusto_bytes(conteudo: bytes):
    """Leitura para uploads (fallback na nuvem)."""
    for sep in [';', ',']:
        for enc in ['utf-8', 'latin1']:
            try:
                df = pd.read_csv(io.BytesIO(conteudo), sep=sep, encoding=enc)
                return df, sep, enc
            except Exception:
                pass
    raise ValueError("Não consegui ler o CSV (separador/codificação).")

# =============================================================================
# Sidebar — datas e parâmetros
# =============================================================================
with st.sidebar:
    st.header("⚙️ Configurações")

    data_corte = st.date_input("Data de corte do treino", value=datetime.today().date())
    data_fim_prev = st.date_input("Data fim da previsão", value=(datetime.today() + timedelta(days=2)).date())
    if data_fim_prev <= data_corte:
        st.warning("A data fim da previsão deve ser maior que a data de corte.")

    st.divider()
    st.markdown("**Arquivos padrão (relativos ao repositório):**")
    st.code(os.path.abspath(CONFIG['ARQUIVO_TOTAL']))
    st.code(os.path.abspath(CONFIG['ARQUIVO_SEGMENTOS']))

    st.divider()
    usar_log = st.toggle("Usar log1p", value=CONFIG['USAR_LOG'])
    aplicar_trava = st.toggle("Aplicar trava de máximos", value=CONFIG['APLICAR_TRAVA_MAXIMA'])
    trava_percentil = st.slider("Percentil da trava", 0.90, 0.999, CONFIG['TRAVA_PERCENTIL'], 0.005)
    conf = st.slider("Nível de confiança (exibição)", 0.60, 0.99, CONFIG['CONFIDENCE_INTERVAL'], 0.01)

    btn_rodar = st.button("▶️ Rodar previsão", type="primary", use_container_width=True)

st.title("🔮 Forecast Diário com SARIMAX — Leitura Local (PX)")
st.caption("Lendo ./data/* do repositório. Se não existir, o app solicita upload e segue normalmente.")

# =============================================================================
# Execução
# =============================================================================
if btn_rodar:
    # 1) Tenta ler pelos caminhos relativos (repo/local)
    try:
        df_total_raw, sep_t, enc_t = carregar_csv_local(CONFIG['ARQUIVO_TOTAL'])
        df_seg_raw,   sep_s, enc_s = carregar_csv_local(CONFIG['ARQUIVO_SEGMENTOS'])
        origem_dados = "repositório (./data)"
    except FileNotFoundError:
        # 2) Fallback de upload — útil na nuvem se você não versionou os CSVs
        st.warning("Arquivos padrão não encontrados em ./data/. Envie os CSVs abaixo para continuar.")
        up_total = st.file_uploader("Total (dados_para_teste.csv)", type=["csv"])
        up_seg   = st.file_uploader("Segmentos (Entrega diária por cavaleiro.csv)", type=["csv"])
        if not up_total or not up_seg:
            st.stop()
        df_total_raw, sep_t, enc_t = carregar_csv_robusto_bytes(up_total.getvalue())
        df_seg_raw,   sep_s, enc_s = carregar_csv_robusto_bytes(up_seg.getvalue())
        origem_dados = "upload"

    # Normalização e detecções
    df_total_raw = normalizar_colunas(df_total_raw)
    df_seg_raw = normalizar_colunas(df_seg_raw)

    col_data_total = detectar_coluna_data(df_total_raw)
    col_data_seg = detectar_coluna_data(df_seg_raw)
    if not col_data_total or not col_data_seg:
        st.error("Não achei coluna de data (contendo 'DIA' ou 'DATA') em um dos arquivos.")
        st.stop()

    col_valor_total = detectar_coluna_valor_total(df_total_raw)
    if not col_valor_total:
        st.error("Não achei coluna de valor no arquivo TOTAL (ex.: VENDA, VENDIDOS, VALOR, QTDE, QTD).")
        st.stop()

    # Limpeza numérica
    df_total_raw[col_valor_total] = limpar_numerico_robusto(df_total_raw[col_valor_total])

    # Features e index
    df_total = aplicar_features_basicas(df_total_raw, col_data_total)
    df_seg = aplicar_features_basicas(df_seg_raw, col_data_seg)

    # Datas e horizonte futuro
    data_corte_treino = pd.to_datetime(data_corte)
    data_fim_previsao = pd.to_datetime(data_fim_prev)
    datas_futuras = pd.date_range(start=data_corte_treino + pd.Timedelta(days=1),
                                  end=data_fim_previsao, freq='D')
    if len(datas_futuras) == 0:
        st.warning("Nada a prever (verifique as datas).")
        st.stop()

    cols_exog = ['IS_WEEKEND', 'IS_MONTH_START', 'IS_HOLIDAY', 'IS_EVENT']
    exog_futuro = preparar_exogenas_index(datas_futuras)

    # Treino TOTAL
    train_tot = df_total.loc[:data_corte_treino]
    if train_tot.empty:
        st.error("Não há dados de treino até a data de corte no TOTAL.")
        st.stop()

    st.subheader("📊 Parâmetros de execução")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Origem dos dados", origem_dados)
    c2.metric("Confiança (exibição)", f"{int(conf*100)}%")
    c3.metric("Log1p", "Sim" if usar_log else "Não")
    c4.metric("Trava", f"{'On' if aplicar_trava else 'Off'} @ p{trava_percentil:.3f}")
    c5.metric("Dias a prever", len(datas_futuras))

    with st.spinner("Treinando TOTAL..."):
        exog_train_tot = train_tot[cols_exog] if set(cols_exog).issubset(train_tot.columns) else preparar_exogenas_index(train_tot.index)
        pred_total = treinar_e_prever(train_tot[col_valor_total], exog_train_tot, exog_futuro,
                                      usar_log=usar_log, aplicar_trava=aplicar_trava,
                                      trava_percentil=trava_percentil, nome="TOTAL")

    # Treino Segmentos
    resultados_seg = pd.DataFrame(index=datas_futuras)
    cols_seg_candidatas = [c for c in df_seg.columns if c not in cols_exog and c != col_data_seg]
    cols_seg = []
    for c in cols_seg_candidatas:
        if not pd.api.types.is_numeric_dtype(df_seg[c]):
            df_seg[c] = limpar_numerico_robusto(df_seg[c])
        cols_seg.append(c)

    with st.spinner("Treinando Segmentos..."):
        for seg in cols_seg:
            train_s = df_seg.loc[:data_corte_treino].copy()
            exog_train_s = train_s[cols_exog] if set(cols_exog).issubset(train_s.columns) else preparar_exogenas_index(train_s.index)
            pred_s = treinar_e_prever(train_s[seg], exog_train_s, exog_futuro,
                                      usar_log=usar_log, aplicar_trava=aplicar_trava,
                                      trava_percentil=trava_percentil, nome=seg)
            if pred_s is not None:
                resultados_seg[seg] = pred_s

    # Consolidação
    df_final = pd.DataFrame(index=datas_futuras)
    df_final['TOTAL_PREVISTO'] = pred_total
    if not resultados_seg.empty:
        df_final = df_final.join(resultados_seg, how='left')
    soma_cols_seg = [c for c in df_final.columns if c != 'TOTAL_PREVISTO']
    df_final['SOMA_SEGMENTOS'] = df_final[soma_cols_seg].sum(axis=1) if soma_cols_seg else 0.0
    df_final.index.name = 'DATA'

    # Fechamento (mês da data de corte)
    vendas_realizadas = df_total.loc[
        (df_total.index.month == data_corte_treino.month) & (df_total.index.year == data_corte_treino.year),
        col_valor_total
    ].sum()
    vendas_previstas_restante = float(df_final['TOTAL_PREVISTO'].sum())
    fechamento_projetado = vendas_realizadas + vendas_previstas_restante

    # Exibição
    st.subheader("💰 Projeção de Fechamento")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Realizado (mês de {data_corte_treino.strftime('%m/%Y')})", f"{vendas_realizadas:,.2f}")
    c2.metric("Previsto (período futuro)", f"{vendas_previstas_restante:,.2f}")
    c3.metric("Total Estimado", f"{fechamento_projetado:,.2f}")

    st.subheader("📈 Gráfico — Total Previsto (diário)")
    st.line_chart(df_final['TOTAL_PREVISTO'])

    if not resultados_seg.empty:
        st.subheader("📈 Gráfico — Soma Segmentos vs Total")
        st.line_chart(df_final[['TOTAL_PREVISTO', 'SOMA_SEGMENTOS']])

        with st.expander("Ver séries por segmento"):
            st.dataframe(df_final.drop(columns=['SOMA_SEGMENTOS']))

    # Download CSV
    nome_arquivo = f"Forecast_de_{data_corte_treino.strftime('%d-%m')}_ate_{data_fim_previsao.strftime('%d-%m-%Y')}.csv"
    csv_bytes = df_final.to_csv(sep=';', decimal=',', encoding='utf-8-sig').encode('utf-8-sig')
    st.download_button("💾 Baixar CSV", data=csv_bytes, file_name=nome_arquivo, mime="text/csv")

    # Debug
    with st.expander("Detalhes técnicos / Debug"):
        st.write("TOTAL (esperado):", os.path.abspath(CONFIG['ARQUIVO_TOTAL']), " | sep/enc:", sep_t, enc_t)
        st.write("SEGMENTOS (esperado):", os.path.abspath(CONFIG['ARQUIVO_SEGMENTOS']), " | sep/enc:", sep_s, enc_s)
        st.write("Coluna de data (TOTAL):", col_data_total)
        st.write("Coluna de valor (TOTAL):", col_valor_total)
        st.write("Coluna de data (SEGMENTOS):", col_data_seg)
        st.write("Segmentos detectados:", cols_seg)
        st.write("Intervalo futuro:", f"{df_final.index[0].date()} → {df_final.index[-1].date()}")
        st.dataframe(df_final.head(10))

else:
    st.info("Use ./data/ no repositório para os CSVs (ou faça upload quando solicitado) e clique **Rodar previsão**.")
