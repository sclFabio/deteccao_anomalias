#### CÓDIGO FEITO PARA RODAR NA VPS DA SCL, USA O POSTGRES.


# ============================
# Bibliotecas
# ============================
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pyodbc
from sklearn.neighbors import LocalOutlierFactor
from lightgbm import LGBMRegressor
from skforecast.direct import ForecasterDirect
from skforecast.preprocessing import RollingFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
import time

# ============================
# Configuração PostgreSQL
# ============================
PG_USER = "postgres"
PG_PASSWORD = quote_plus("scl@0102")  # trata caracteres especiais
PG_HOST = "93.127.212.75"
PG_PORT = "5432"
PG_DB = "SAMAE-SBS-PREVISAO"

engine = create_engine(
    f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
)


#tabela = 'EAT019'
tabela = 'EAT028'
coluna_valor = 'FTS001'
estacao = tabela  


data_ref = pd.Timestamp('2025-09-10 00:00:00')
data_limite = '2025-09-10'

# ============================
# Funções
# ============================


def conectar_sql(tabela, colunas, servidor, database, usuario, senha, max_retries=3):
    colunas_sql = ", ".join(f"[{c}]" for c in colunas)
    tentativa = 0

    while tentativa <= max_retries:
        try:
            conn = pyodbc.connect(
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={servidor};DATABASE={database};"
                f"UID={usuario};PWD={senha}",
                timeout=10  
            )
            query = f"SELECT {colunas_sql} FROM {database}.dbo.{tabela}"
            data = pd.read_sql(query, conn, parse_dates=[colunas[0]])
            conn.close()
            return data

        except Exception as e:
            print(f"Tentativa {tentativa+1} falhou: {e}")
            tentativa += 1

            if tentativa > max_retries:
                print("Erro de conexão persistente. Encerrando o programa.")
                raise e  
            
            wait_time = 2 ** tentativa
            print(f"Tentando novamente em {wait_time} segundos...")
            time.sleep(wait_time)

def preparar_dados(data, coluna_valor, data_limite):
    data = data[pd.to_datetime(data['E3TimeStamp'], errors='coerce').notna()]
    data['E3TimeStamp'] = pd.to_datetime(data['E3TimeStamp'])
    data = data.sort_values('E3TimeStamp').set_index('E3TimeStamp')

    # Remove outliers extremos com IQR
    Q1 = data[coluna_valor].quantile(0.25)
    Q3 = data[coluna_valor].quantile(0.75)
    IQR = Q3 - Q1
    data = data[(data[coluna_valor] >= 0) & (data[coluna_valor] <= Q3 + 1.5*IQR)]

    # Resample e interpolação
    data = data.resample('30min').mean().interpolate()
    data = data.loc[data.index < pd.to_datetime(data_limite)]
    data['Anomalia'] = 0
    data['Anomalia_final'] = 0

    # Perfil diário médio
    profile = data.groupby([data.index.weekday, data.index.hour])[coluna_valor].mean()
    data_inicio = pd.Timestamp('2024-01-02 00:00:00')
    inicio = (data_inicio - timedelta(days=1)).normalize()  
    fim = data_ref.normalize()  
    data = data[(data.index >= inicio) & (data.index < fim)].copy()
    # Features externas
    exog = pd.DataFrame({
        'hour': data.index.hour,
        'weekday': data.index.weekday,
        'is_weekend': (data.index.weekday >= 5).astype(int)
    }, index=data.index)
    return data, profile, exog

def forecast(data, exog, coluna_valor, steps):
    data_train = data[:-steps]
    data_test = data[-steps:]
    exog_train = exog.iloc[:-steps]
    exog_test = exog.iloc[-steps:]

    forecaster = ForecasterDirect(
        regressor=LGBMRegressor(
            learning_rate=0.03,
            n_estimators=500,
            num_leaves=31,
            max_depth=-1,
            random_state=123,
            verbose=-1
        ),
        lags=24,
        steps=steps,
        window_features=RollingFeatures(
            stats=['mean', 'std', 'max', 'min'],
            window_sizes=[6, 12, 24, 48]
        )
    )
    forecaster.fit(y=data_train[coluna_valor], exog=exog_train)
    preds = forecaster.predict(steps=steps, exog=exog_test)
    preds.index = data_test.index

    mae = mean_absolute_error(data_test[coluna_valor], preds)
    print("Mean Absolute Error")
    print(mae)
    rmse = mean_squared_error(data_test[coluna_valor], preds)
    print(rmse)
    
    

    return preds, mae, rmse

def detectar_anomalias(data, coluna_valor):
    # Configura LOF
    lof = LocalOutlierFactor(n_neighbors=5, contamination=0.01)
    
    # Detecta anomalias usando a coluna passada
    labels = lof.fit_predict(data[[coluna_valor]])
    data['lof_anomalia'] = (labels == -1).astype(int)
    
    # Marca a coluna de anomalia principal
    data['Anomalia'] = (data['lof_anomalia'] == 1).astype(int)

    # ========= comparação com os 2 anteriores =========
    vals = data[coluna_valor].to_numpy()
    cond = np.zeros(len(vals), dtype=bool)
    cond[2:] = (vals[2:] > vals[1:-1]) & (vals[2:] > vals[:-2])  # pico p/ cima
    cond_manter = (data['Anomalia'].to_numpy() == 1) & cond
    # ==================================================

    # Zera as anomalias que não atendem a regra do pico
    data.loc[data['Anomalia'] == 1, 'Anomalia'] = 0
    data.loc[cond_manter, 'Anomalia'] = 1

    # Inicializa a coluna final a partir da "limpa"
    data['Anomalia_final'] = data['Anomalia'].copy()

    # Extensão das anomalias de 00h-06h para próximas 24h
    for ts in data.index[data['Anomalia'] == 1]:
        if 0 <= ts.hour < 6:
            fim = ts + timedelta(hours=24)
            data.loc[(data.index >= ts) & (data.index <= fim), 'Anomalia_final'] = 1
    #print(data.info)        
    inicio = (data_ref - timedelta(days=1)).normalize()  # 00:00 do dia anterior
    fim = data_ref.normalize()  # 00:00 do dia de referência
    anomalia = data[(data.index >= inicio) & (data.index < fim)].copy()
    
    anomalia = anomalia[anomalia["Anomalia"] == 1] 
    
    if anomalia.empty:
        return None, data
    anomalia = anomalia.iloc[0]
    

    return anomalia, data


def buscar_referencia(engine, estacao):
    """Busca a última anomalia de referência ativa no banco"""
    query = text(f"""
        SELECT * FROM anomalias_referencia
        WHERE estacao = :estacao AND anomalia_ativa = TRUE
        ORDER BY timestamp DESC
        LIMIT 1
    """)
    df_ref = pd.read_sql(query, engine, params={"estacao": estacao})
    if df_ref.empty:
        return None
    return df_ref.iloc[0]

def atualizar_anomalia(data, coluna_valor, engine, estacao, data_limite):
    """
    Atualiza Anomalia_final linha a linha:
    - Liga se valor >= ref_val e anomalia não foi encerrada
    - Desliga se valor < ref_val
    - Uma vez encerrada, não volta a ligar
    - Atualiza anomalia_ativa no banco se encerrada
    """
    data_limite = pd.to_datetime(data_limite)
    inicio = (data_limite - timedelta(days=1)).normalize()
    fim = data_limite.normalize()
    #data = data[(data.index >= inicio) & (data.index < fim)].copy()

    ref = buscar_referencia(engine, estacao)
    print("Referencia Ativa a ser considerada")
    print(ref)
    if ref is None:
        print("Nenhuma referência ativa encontrada.")
        return data

    ref_val = ref['valor'] * 0.995
    
    ref_id = ref['id']
    data_min = pd.to_datetime(ref['timestamp'])

    estados = []
    encerrada = False  # flag para impedir reativação

    for ts, row in data.iterrows():
        if ts >= data_min and not encerrada:
            if row[coluna_valor] >= ref_val:
                estado = 1
            else:
                estado = 0
                encerrada = True
                # Atualiza banco na primeira vez que encerra
                if ref['anomalia_ativa']:
                    with engine.begin() as conn:
                        update_query = text("""
                            UPDATE public.anomalias_referencia
                            SET anomalia_ativa = FALSE
                            WHERE id = :id
                        """)
                        conn.execute(update_query, {"id": int(ref_id)})
                        ref['anomalia_ativa'] = False
                        print(f"Anomalia encerrada e desativada no banco (id={ref_id})")
        else:
            estado = 0

        estados.append(estado)

    data['Anomalia_final'] = estados
    
    
    return data

# --- Funções de persistência ---
def criar_referencia_ativa(df_series, estacao):
    print("Anomalia a ser inserida.")
    
        # Pega os valores da Series
    timestamp       = df_series.name              
    valor_ft        = df_series["FTS001"]
    anomalia_final  = df_series["Anomalia_final"]
   
    ref = {
        "estacao": estacao,
        "timestamp": timestamp,
        "valor": valor_ft,
        "anomalia_ativa": (anomalia_final == 1)
    }
    return ref if ref["anomalia_ativa"] else None

def salvar_anomalia_referencia(engine, estacao, ref):
    """
    Salva uma nova anomalia de referência no banco somente se:
    - ref não for None
    - não existir outra anomalia ativa para a mesma estação
    """
    if ref is None or not ref.get("anomalia_ativa", False):
        print("Nenhuma anomalia ativa para salvar.")
        return

    # Verifica se já existe anomalia ativa para essa estação
    query = text("""
        SELECT id FROM public.anomalias_referencia
        WHERE estacao = :estacao AND anomalia_ativa = TRUE
        LIMIT 1
    """)
    with engine.connect() as conn:
        existing = conn.execute(query, {"estacao": estacao}).fetchone()

    if existing:
        print(f"Já existe anomalia ativa para a estação {estacao} (id={existing.id}). Nova não será salva.")
        return
    print(ref)
    # Se não existe, salva a nova
    df_ref = pd.DataFrame([{
        "estacao": estacao,
        "timestamp": ref["timestamp"],
        "valor": ref["valor"],
        "anomalia_ativa": True
    }])
    df_ref.to_sql("anomalias_referencia", engine, if_exists="append", index=False)
    print("Anomalia de referência salva com sucesso.")

def salvar_serie_temporal(engine, estacao, df_series):
    
    df_series["anomalia_detectada"] = df_series["anomalia_detectada"].astype(bool)
    df_series["anomalia_persistente"] = df_series["anomalia_persistente"].astype(bool)
    print("Conteúdo a ser salvo na DB")
    print(df_series)

    df_series.to_sql(f"serie_temporal_{estacao}", engine, if_exists="append", index=False)
    print(f"{len(df_series)} registros de série temporal inseridos.")

# ============================
# Execução principal
# ============================

# Conexão e dados
data = conectar_sql(
    tabela=tabela,
    colunas=["E3TimeStamp", "FTS001"],
    servidor='172.16.101.70',
    database='DBArea',
    usuario='sa',
    senha='SMS@2104-056'
)

# Preparação
data, profile, exog = preparar_dados(data, coluna_valor=coluna_valor, data_limite=data_limite)
steps = 48 * 10
preds, mae, rmse = forecast(data, exog, coluna_valor, steps=steps)


# Detecta anomalias
anomalia ,data = detectar_anomalias(data, coluna_valor=coluna_valor)

if anomalia is not None:
    nova_ref = criar_referencia_ativa(anomalia, estacao)
    #salvar_anomalia_referencia(engine, estacao, nova_ref)
else:
    print("Nenhuma anomalia detectada. Nenhuma referência será salva.")





# Atualiza referência (persistência com base no banco)
data = atualizar_anomalia(data, coluna_valor=coluna_valor, engine=engine, estacao=estacao, data_limite=data_limite)


# Prepara DataFrame para salvar série temporal
df_series = data.reset_index().rename(columns={"E3TimeStamp": "timestamp"})
df_series["valor_ft"] = df_series[coluna_valor]
df_series = df_series.set_index("timestamp")
df_series["previsao"] = preds
df_series = df_series.reset_index()
df_series = df_series[["timestamp", "valor_ft", "previsao", "Anomalia", "Anomalia_final"]].copy()
df_series = df_series.rename(columns={
    "Anomalia": "anomalia_detectada",
    "Anomalia_final": "anomalia_persistente"
})
#print(data)
# Filtra apenas o dia anterior à data de referência
inicio = (data_ref - timedelta(days=1)).normalize()  # 00:00 do dia anterior
fim = data_ref.normalize()  # 00:00 do dia de referência
#df_series = df_series[(df_series["timestamp"] >= inicio) & (df_series["timestamp"] < fim)].copy()



"""nova_ref = criar_referencia_ativa(df_series, estacao)
salvar_anomalia_referencia(engine, estacao, nova_ref)"""
#salvar_serie_temporal(engine, estacao, df_series)


print(df_series)
def plotar_grafico(data, profile, coluna_valor, engine, estacao):
    ref = buscar_referencia(engine, estacao)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["valor_ft"], mode='lines', name='Histórico'))
    fig.add_trace(go.Scatter(x=data.index, y=data["previsao"], mode='lines', name='Previsao'))
    fig.add_trace(go.Scatter(
        x=data.loc[data['anomalia_persistente'] == 1].index,
        y=data.loc[data['anomalia_persistente'] == 1, "valor_ft"],
        mode='markers', name='Anomalia', marker=dict(color='orange', size=8)
    ))
    if ref is not None:
        inicio_anomalia = pd.Timestamp(ref['timestamp'])
        data_ativo = data.loc[data.index >= inicio_anomalia]
        fig.add_trace(go.Scatter(
            x=data_ativo.index,
            y=[ref['valor']] * len(data_ativo),
            mode='lines', name='Anomalia Ativa', line=dict(color='red', width=2, dash='dash')
        ))
    fig.update_layout(title="Série temporal com anomalias", xaxis_title="Data", yaxis_title=coluna_valor, template="plotly_dark")
    return fig
profile.to_csv(f"/root/venv/perfil_diario_{tabela}.csv")
df_series.to_csv(f"/root/venv/{tabela}.csv")
fig = plotar_grafico(df_series.set_index("timestamp"), profile, coluna_valor, engine, estacao)
fig.write_html(f"/root/venv/previsao_{tabela}_{data_limite}.html")
