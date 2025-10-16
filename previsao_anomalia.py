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
t0 = time.time()
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


# ============================
# Configuração MSSQL
# ============================

# CRIAR TABELA NO BANCO CASO NÃO EXISTA (FAZER DIRETO NO MSSQL):

"""
CREATE TABLE "ETA001_FTS001"
(
    "E3TimeStamp" datetime2,
    valor_ft double precision,
    previsao text,
    anomalia_detectada bit,
    anomalia_persistente bit
)
"""


driver = 'ODBC Driver 17 for SQL Server'
server = '172.16.101.70'
database = "DB_AlarmeAnomalia"
username = 'sa'
password = 'SMS@2104-056'

conn_str_mssql = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}'
sql_alchemy_conn_str = f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver}'


# ============================
# Funções
# ============================


def conectar_sql(tabela, colunas, servidor, database, usuario, senha, comando, max_retries=3):
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
            if comando == 'SELECT':
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
    

    # Features externas
    exog = pd.DataFrame({
        'hour': data.index.hour,
        'weekday': data.index.weekday,
        'is_weekend': (data.index.weekday >= 5).astype(int)
    }, index=data.index)
    #print(data.describe())
    return data, profile, exog

def forecast(data, exog, coluna_valor, steps):
    data_train = data[:-steps]
    data_test = data[-steps:]
    exog_train = exog.iloc[:-steps]
    exog_test = exog.iloc[-steps:]
    #lags 24 = 12 horas
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
    rmse = mean_squared_error(data_test[coluna_valor], preds)

    

    return preds, mae, rmse

def detectar_anomalias(data, coluna_valor, usar_previsao=False):
    # Configura LOF
    lof = LocalOutlierFactor(n_neighbors=5, contamination=0.05)
    
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

    # --- Critério extra: erro em relação à previsão ---
    if usar_previsao and "previsao" in data.columns and data["previsao"].notna().any():
        erros = (data[coluna_valor] - data["previsao"]).abs()
        print("Erros")
        print(erros)
        
        limiar = erros.mean() + 3 * erros.std()
        print("Limiar")
        print(limiar)
        data["erro_previsao"] = erros
        data.loc[erros > limiar, "Anomalia"] = 1

    # --- Extensão madrugada (00h-06h → +24h) ---
    data['Anomalia_final'] = data['Anomalia'].copy()
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
    print(anomalia)
    print(data.tail(48))
    

    return anomalia, data


def buscar_referencia(estacao):
    """Busca a última anomalia de referência ativa no banco (MSSQL)"""
    conn = pyodbc.connect(
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER=172.16.101.70;DATABASE=DB_AlarmeAnomalia;"
        f"UID=sa;PWD=SMS@2104-056",
        timeout=10
    )
    cursor = conn.cursor()

    query = """
        SELECT TOP 1 id, estacao, data_evento, valor, anomalia_ativa
        FROM dbo.anomalias_referencia
        WHERE estacao = ? AND anomalia_ativa = 1
        ORDER BY data_evento DESC
    """
    cursor.execute(query, (estacao,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "id": row[0],
        "estacao": row[1],
        "timestamp": row[2],  
        "valor": row[3],
        "anomalia_ativa": row[4]
    }

def atualizar_anomalia(data, coluna_valor, estacao, data_limite):
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
    data = data[(data.index >= inicio) & (data.index < fim)].copy()

    ref = buscar_referencia(estacao)
    print(f"Referência encontrada: {ref}")

    if ref is None:
        print("Nenhuma referência ativa encontrada.")
        return data

    ref_val = float(ref['valor']) * 0.995
    ref_id = ref['id']
    data_min = pd.to_datetime(ref['timestamp'])

    estados = []
    encerrada = False 

    for ts, row in data.iterrows():
        if ts >= data_min and not encerrada:
            if row[coluna_valor] >= ref_val:
                estado = 1
            else:
                estado = 0
                encerrada = True
                # Atualiza banco na primeira vez que encerra
                if ref['anomalia_ativa']:
                    conn = pyodbc.connect(
                        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                        f"SERVER=172.16.101.70;DATABASE=DB_AlarmeAnomalia;"
                        f"UID=sa;PWD=SMS@2104-056",
                        timeout=10
                    )
                    cursor = conn.cursor()
                    update_query = """
                        UPDATE dbo.anomalias_referencia
                        SET anomalia_ativa = 0
                        WHERE id = ?
                    """
                    cursor.execute(update_query, (ref_id,))
                    conn.commit()
                    conn.close()
                    ref['anomalia_ativa'] = False
                    print(f"Anomalia encerrada e desativada no banco (id={ref_id})")
        else:
            estado = 0

        estados.append(estado)

    data['Anomalia_final'] = estados
    return data

# --- Funções de persistência ---
def criar_referencia_ativa(df_series, estacao):

    
    # Pega os valores da Series
    
    timestamp = df_series.name              
    valor_ft = df_series["FTS001"]
    anomalia_final = df_series["Anomalia_final"]
   
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
    conn = pyodbc.connect(
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={'172.16.101.70'};DATABASE={"DB_AlarmeAnomalia"};"
                f"UID={'sa'};PWD={'SMS@2104-056'}",
                timeout=10  
    )
    cursor = conn.cursor()

    
    query_check = """
        SELECT TOP 1 id 
        FROM dbo.anomalias_referencia
        WHERE estacao = ? AND anomalia_ativa = 1
    """
    cursor.execute(query_check, (estacao,))
    existing = cursor.fetchone()

    if existing:
        print(f"Já existe anomalia ativa para a estação {estacao} (id={existing[0]}). Nova não será salva.")
        conn.close()
        return

    
    query_insert = """
        INSERT INTO dbo.anomalias_referencia (estacao, valor, data_evento, anomalia_ativa)
        VALUES (?, ?, ?, 1)
    """
    print(f"Referencia = {ref}")
    cursor.execute(query_insert, (ref["estacao"], ref["valor"], ref["timestamp"]))
    conn.commit()
    conn.close()

    print(f"Anomalia salva com sucesso para estação {estacao}.")



def salvar_serie_temporal(conn_str, estacao, df_series):
    df_series["anomalia_detectada"] = df_series["anomalia_detectada"].astype(bool)
    df_series["anomalia_persistente"] = df_series["anomalia_persistente"].astype(bool)

    conn = pyodbc.connect(
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={'172.16.101.70'};DATABASE={"DB_AlarmeAnomalia"};"
                f"UID={'sa'};PWD={'SMS@2104-056'}",
                timeout=10  
    )
    cursor = conn.cursor()

    table_name = f"serie_temporal_{estacao}"
    

    query_insert = f"""
        INSERT INTO dbo.{table_name} (E3TimeStamp, valor_ft, previsao, anomalia_detectada, anomalia_persistente)
        VALUES (?, ?, ?, ?, ?)
    """
    print(df_series)
    rows = [
        (
            row["timestamp"].to_pydatetime(),
            float(row["valor_ft"]),
            #Trocar para float
            str(row["previsao"]),
            bool(row["anomalia_detectada"]),
            bool(row["anomalia_persistente"]),
        )
        for _, row in df_series.iterrows()
    ]

    cursor.executemany(query_insert, rows)
    conn.commit()
    conn.close()

    print(f"{len(df_series)} registros de série temporal inseridos em {table_name}.")


def plotar_grafico(data, profile, coluna_valor, engine, estacao):
    ref = buscar_referencia(engine, estacao)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["valor_ft"], mode='lines', name='Histórico'))
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

# ============================
# Execução principal
# ============================

tabelas = ['ETA001', 'EAT004', 'EAT006', 'EAT007', 'EAT017', 'EAT019', 'EAT021', 'EAT024', 'EAT025', 'EAT027', 'EAT028', 'EAT030', 'EAT032']
coluna_valor = 'FTS001'


#Trocar a frequência para 12h? 6h? Horário?
#data_ref = pd.Timestamp('2025-09-06 00:00:00')
#data_limite = '2025-09-06'
data_ref = pd.to_datetime(datetime.now().date())
data_limite = pd.to_datetime(datetime.now().date())


for tabela in tabelas:
    
    estacao = tabela
    
    print(f"Processando dados da estação: {estacao}")
    
    data = conectar_sql(
    tabela=tabela,
    colunas=["E3TimeStamp", "FTS001"],
    servidor='172.16.101.70',
    database='DBArea',
    usuario='sa',
    senha='SMS@2104-056',
    comando='SELECT'
)

    # Preparação
    data, profile, exog = preparar_dados(data, coluna_valor=coluna_valor, data_limite=data_limite)
    steps=48
    preds, mae, rmse = forecast(data, exog, coluna_valor, steps=steps)


    # Detecta anomalias
    anomalia ,data = detectar_anomalias(data, coluna_valor=coluna_valor)

    if anomalia is not None:
        nova_ref = criar_referencia_ativa(anomalia, estacao)
        #salvar_anomalia_referencia(engine, estacao, nova_ref)
        salvar_anomalia_referencia('banco_samae', estacao, nova_ref)
    else:
        print("Nenhuma anomalia detectada. Nenhuma referência será salva.")
    

    # Atualiza referência (persistência com base no banco)
    #data = atualizar_anomalia(data, coluna_valor=coluna_valor, engine=engine, estacao=estacao, data_limite=data_limite)
    data = atualizar_anomalia(data, coluna_valor=coluna_valor, estacao=estacao, data_limite=data_limite)


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
    df_series = df_series[(df_series["timestamp"] >= inicio) & (df_series["timestamp"] < fim)].copy()

    #print(df_series)

    #nova_ref = criar_referencia_ativa(df_series, estacao)
    #salvar_anomalia_referencia(engine, estacao, nova_ref)
    #salvar_serie_temporal(engine, estacao, df_series)
    salvar_serie_temporal('banco_samae', estacao, df_series)


    #print(data)


    """profile.to_csv(f"/root/venv/perfil_diario_{tabela}.csv")"""
    """fig = plotar_grafico(df_series.set_index("timestamp"), profile, coluna_valor, engine, estacao)
    fig.write_html(f"/root/venv/previsao_{tabela}_{data_limite}.html")"""

#Mostrar o tempo de execução do programa.
t1 = time.time()
print(f"Tempo de execução: {t1 - t0:.4f} segundos")
