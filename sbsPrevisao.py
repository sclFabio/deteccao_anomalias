# ============================
# Bibliotecas
# ============================
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pyodbc
from lightgbm import LGBMRegressor
from skforecast.direct import ForecasterDirect
from skforecast.preprocessing import RollingFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
import time

t0 = time.time()

# ============================
# Configuração PostgreSQL
# ============================
PG_USER = "postgres"
PG_PASSWORD = quote_plus("scl@0102")
PG_HOST = "93.127.212.75"
PG_PORT = "5432"
PG_DB = "SAMAE-SBS-PREVISAO"

engine = create_engine(
    f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
)

# ============================
# Intervalo de previsão (últimas 6h concluídas)
# ============================
now = datetime.now()
hora_atual = now.hour

if 0 <= hora_atual < 6:
    data_ref = pd.to_datetime(now.date() - timedelta(days=1)) + timedelta(hours=18)
    data_limite = pd.to_datetime(now.date())
    LIMIAR_PCT = 0.10
    LIMIAR_ABS = 10.0
    
    
elif 6 <= hora_atual < 12:
    data_ref = pd.to_datetime(now.date())
    data_limite = pd.to_datetime(now.date()) + timedelta(hours=6)
    LIMIAR_PCT = 0.15
    LIMIAR_ABS = 15.0
    
elif 12 <= hora_atual < 18:
    data_ref = pd.to_datetime(now.date()) + timedelta(hours=6)
    data_limite = pd.to_datetime(now.date()) + timedelta(hours=12)
    LIMIAR_PCT = 0.20
    LIMIAR_ABS = 20.0
else:
    data_ref = pd.to_datetime(now.date()) + timedelta(hours=12)
    data_limite = pd.to_datetime(now.date()) + timedelta(hours=18)
    LIMIAR_PCT = 0.20
    LIMIAR_ABS = 20.0

print(f"Previsões para o intervalo: {data_ref} → {data_limite}")

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

def preparar_dados(data, coluna_valor, data_limite, estacao):
    # 1. Limpeza básica inicial
    data = data[pd.to_datetime(data['E3TimeStamp'], errors='coerce').notna()]
    data['E3TimeStamp'] = pd.to_datetime(data['E3TimeStamp'])
    data = data.sort_values('E3TimeStamp').set_index('E3TimeStamp')
    data = data.loc[data.index <= pd.to_datetime(data_limite)]
    data_bruto = data.copy()
    data_bruto = data_bruto.resample('30min').mean().interpolate()
    # Remove valores negativos e outliers com IQR
    Q1 = data[coluna_valor].quantile(0.25)
    Q3 = data[coluna_valor].quantile(0.75)
    IQR = Q3 - Q1
    data = data[(data[coluna_valor] >= 0) & (data[coluna_valor] <= Q3 + 1.5 * IQR)]

    # 2. Identificar dias com anomalia
    dias_anomalia = obter_dias_com_anomalia(estacao)
    print(f"Dias com anomalia para {estacao}: {sorted(dias_anomalia)}")

    # 3. Criar máscara de dados INVÁLIDOS (só para limpeza)
    mask_anomalia_dia = np.isin(data.index.date, list(dias_anomalia))
    mask_zero = data[coluna_valor] <= 0.1
    mask_invalido = mask_anomalia_dia | mask_zero

    # 4. Criar cópia para o dataset "limpo"
    data_limpo = data.copy()

    # 5. Calcular perfil diário usando APENAS dias NORMAIS
    dados_normais = data_limpo[~np.isin(data_limpo.index.date, list(dias_anomalia))]
    dados_normais = dados_normais[dados_normais[coluna_valor] > 1.0]

    if not dados_normais.empty:
        perfil = dados_normais.groupby([dados_normais.index.weekday, dados_normais.index.hour])[coluna_valor].mean()
    else:
        perfil = data.groupby([data.index.weekday, data.index.hour])[coluna_valor].mean()

    # 6. Substituir valores inválidos pelo perfil (só em data_limpo)
    if mask_invalido.any():
        def substituir_por_perfil(ts, valor_atual):
            if mask_invalido.loc[ts]:
                return perfil.get((ts.weekday(), ts.hour), valor_atual)
            return valor_atual

        data_limpo[coluna_valor] = [
            substituir_por_perfil(ts, valor) 
            for ts, valor in zip(data_limpo.index, data_limpo[coluna_valor])
        ]

    # 7. RESAMPLE + INTERPOLAÇÃO — aplicar a AMBOS os datasets
    #    Isso garante que tenham os mesmos timestamps
    data_limpo = data_limpo.resample('30min').mean().interpolate()
    


    # 8. Features exógenas (baseadas no índice final)
    exog = pd.DataFrame({
        'hour': data_limpo.index.hour,
        'weekday': data_limpo.index.weekday,
        'is_weekend': (data_limpo.index.weekday >= 5).astype(int)
    }, index=data_limpo.index)
    
    return data_bruto, data_limpo, perfil, exog

def forecast(data, exog, coluna_valor, steps):

    data_train = data[:-steps]
    data_test = data[-steps:]
    exog_train = exog.iloc[:-steps]
    exog_test = exog.iloc[-steps:]
    #lags 24 = 12 horas
    forecaster = ForecasterDirect(
        regressor=LGBMRegressor(
            learning_rate=0.03,
            n_estimators=300,
            num_leaves=20,
            max_depth=-1,
            random_state=123,
            verbose=-1
        ),
        #lags=12,
        lags=504,
        steps=steps,
        window_features=RollingFeatures(
            stats=['mean', 'std', 'max', 'min'],
            #window_sizes=[6, 12, 24, 48]
            window_sizes=[24, 48, 72, 168]
        )
    )
    forecaster.fit(y=data_train[coluna_valor], exog=exog_train)
    preds = forecaster.predict(steps=steps, exog=exog_test)
    preds.index = data_test.index

    mae = mean_absolute_error(data_test[coluna_valor], preds)
    rmse = mean_squared_error(data_test[coluna_valor], preds)

    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    #print(preds)
    return preds, mae, rmse

def salvar_inicio_vazamento(estacao, timestamp, valor_real, previsao):
    """
    Salva o início de um vazamento persistente na tabela MSSQL `anomalias_referencia`.
    Só salva se NÃO houver outra anomalia ativa para a mesma estação.
    Define `correcao_manual = 1` conforme seu modelo.
    """
    # Parâmetros de conexão
    conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=172.16.101.70;"
        "DATABASE=DB_AlarmeAnomalia;"
        "UID=sa;"
        "PWD=SMS@2104-056"
    )

    try:
        conn = pyodbc.connect(conn_str, timeout=10)
        cursor = conn.cursor()

        # 1. Verifica se já existe anomalia ativa para esta estação
        check_query = """
            SELECT TOP 1 id 
            FROM dbo.anomalias_referencia 
            WHERE estacao = ? AND anomalia_ativa = 1
        """
        cursor.execute(check_query, (estacao,))
        if cursor.fetchone():
            print(f"⚠️ Já existe anomalia ativa para {estacao}. Não salvando duplicata.")
            conn.close()
            return

        # 2. Insere novo vazamento
        insert_query = """
            INSERT INTO dbo.anomalias_referencia 
            (estacao, valor, data_evento, anomalia_ativa, correcao_manual)
            VALUES (?, ?, ?, 1, 0)
        """
        cursor.execute(insert_query, (estacao, float(valor_real), timestamp))
        conn.commit()
        conn.close()

        print(f"✅ Vazamento salvo em anomalias_referencia: {estacao} em {timestamp}")

    except Exception as e:
        print(f"❌ Erro ao salvar vazamento no MSSQL: {e}")
        if 'conn' in locals():
            conn.close()


def desativar_vazamento_se_resolvido(estacao, df_intervalo, limiar_abs, limiar_pct):
    """
    Desativa vazamento ativo no MSSQL se todos os valores no intervalo estiverem dentro dos limiares.
    """

    conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=172.16.101.70;"
        "DATABASE=DB_AlarmeAnomalia;"
        "UID=sa;"
        "PWD=SMS@2104-056"
    )

    try:
        conn = pyodbc.connect(conn_str, timeout=10)
        cursor = conn.cursor()

        # 1. Buscar vazamento ativo
        query_ativo = """
            SELECT id FROM dbo.anomalias_referencia
            WHERE estacao = ? AND anomalia_ativa = 1
        """
        cursor.execute(query_ativo, (estacao,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return False  # Nada para desativar

        ref_id = row[0]

        # 2. Verificar se todos os valores do intervalo estão dentro dos limiares
        if df_intervalo.empty:
            conn.close()
            return False

        # Calcular desvios
        df = df_intervalo.copy()
        df = df[df['valor_ft'].notna() & df['previsao'].notna()]
        if df.empty:
            conn.close()
            return False

        erro_abs = df['valor_ft'] - df['previsao']
        erro_pct = erro_abs / df['previsao'].clip(lower=1e-6)

        # Condição: valor real <= previsão + limiares
        dentro_abs = erro_abs <= limiar_abs
        dentro_pct = erro_pct <= limiar_pct
        dentro = dentro_abs & dentro_pct

        # Se TODOS os pontos estão dentro da faixa → desativar
        if dentro.all() and len(dentro) >= 2:  # pelo menos 1h de dados
            update_query = """
                UPDATE dbo.anomalias_referencia
                SET anomalia_ativa = 0
                WHERE id = ?
            """
            cursor.execute(update_query, (ref_id,))
            conn.commit()
            conn.close()
            print(f"✅ Vazamento ID={ref_id} desativado: consumo normalizado em {estacao}.")
            return True

        conn.close()
        return False

    except Exception as e:
        print(f"❌ Erro ao verificar/desativar vazamento: {e}")
        if 'conn' in locals():
            conn.close()
        return False        
def buscar_vazamento_ativo(estacao):
    """
    Retorna os dados do vazamento ativo para a estação, ou None se não houver.
    """

    conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=172.16.101.70;"
        "DATABASE=DB_AlarmeAnomalia;"
        "UID=sa;"
        "PWD=SMS@2104-056"
    )

    try:
        conn = pyodbc.connect(conn_str, timeout=10)
        cursor = conn.cursor()
        query = """
            SELECT data_evento, valor, correcao_manual 
            FROM dbo.anomalias_referencia
            WHERE estacao = ? AND anomalia_ativa = 1
        """
        cursor.execute(query, (estacao,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return None

        ref_id, data_evento, valor, correcao_manual = row 

        if correcao_manual == 1:
            print(f"Anomalia ID={ref_id} corrigida manualmente. Desativando...")
            # Reabrir conexão para atualizar
            conn2 = pyodbc.connect(conn_str, timeout=10)
            cursor2 = conn2.cursor()
            cursor2.execute(
                "UPDATE dbo.anomalias_referencia SET anomalia_ativa = 0 WHERE id = ?",
                (ref_id,)
            )
            conn2.commit()
            conn2.close()
            return None

        conn.close()
        return {
            "id": ref_id,
            "timestamp": data_evento,
            "valor": valor
        }

    except Exception as e:
        print(f"Erro ao buscar/validar vazamento ativo: {e}")
        return None
def salvar_serie_temporal(estacao, df_series):

    df_series["anomalia_detectada"] = df_series["anomalia_detectada"].astype(bool)
    df_series["anomalia_persistente"] = df_series["anomalia_persistente"].astype(bool)
    # Conexão
    conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=172.16.101.70;"
        "DATABASE=DB_AlarmeAnomalia;"
        "UID=sa;"
        "PWD=SMS@2104-056"
    )

    try:
        conn = pyodbc.connect(conn_str, timeout=10)
        cursor = conn.cursor()

        table_name = f"{estacao}_FTSTESTE"

        # Garantir que a tabela existe (opcional, mas útil)
        create_table = f"""
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='{table_name}' AND xtype='U')
        CREATE TABLE dbo.{table_name} (
            E3TimeStamp datetime2,
            valor_ft float,
            previsao float,
            anomalia_detectada bit,
            anomalia_persistente bit
        )
        """
        cursor.execute(create_table)
        # Inserir
        query_insert = f"""
            INSERT INTO dbo.{table_name} 
            (E3TimeStamp, valor_ft, previsao, anomalia_detectada, anomalia_persistente)
            VALUES (?, ?, ?, ?, ?)
        """
        rows = [
        (
            row["timestamp"].to_pydatetime(),
            float(row["valor_ft"]),            
            bool(row["anomalia_detectada"]),
            bool(row["anomalia_persistente"]),
            float(row["previsao"]),
        )
        for _, row in df_series.iterrows()
    ]
        cursor.executemany(query_insert, rows)
        conn.commit()
        conn.close()

        print(f"{len(rows)} registros salvos em {table_name}")

    except Exception as e:
        print(f"Erro ao salvar série temporal: {e}")
        if 'conn' in locals():
            conn.close()

def obter_dias_com_anomalia(estacao):
    """
    Retorna conjunto de datas (datetime.date) que têm anomalia ativa.
    """

    conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=172.16.101.70;"
        "DATABASE=DB_AlarmeAnomalia;"
        "UID=sa;"
        "PWD=SMS@2104-056"
    )

    try:
        conn = pyodbc.connect(conn_str, timeout=10)
        query = """
            SELECT data_evento
            FROM dbo.anomalias_referencia
            WHERE estacao = ? AND anomalia_ativa = 1
        """
        df_anom = pd.read_sql(query, conn, params=[estacao])
        conn.close()

        if df_anom.empty:
            return set()

        df_anom['data_evento'] = pd.to_datetime(df_anom['data_evento'], errors='coerce')
        dias_anomalia = set()
        for ts in df_anom['data_evento'].dropna():
            # Inclui o dia da anomalia (você pode ajustar para +1 dia se quiser)
            dias_anomalia.add(ts.date())
        return dias_anomalia

    except Exception as e:
        print(f"Erro ao buscar dias com anomalia: {e}")
        return set()

# ============================
# Execução principal
# ============================

#tabelas = ['ETA001', 'EAT004', 'EAT006', 'EAT007', 'EAT017', 'EAT019', 'EAT021',
#           'EAT024', 'EAT025', 'EAT027', 'EAT028', 'EAT030', 'EAT032']

tabelas = ['EAT032']
coluna_valor = 'FTS001'



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
    data_bruto ,dados_limpos, profile, exog = preparar_dados(data, coluna_valor=coluna_valor, data_limite=data_limite, estacao=estacao)    
    steps=12
    
    preds, mae, rmse = forecast(dados_limpos, exog, coluna_valor, steps=steps)
    
    valor_real = data_bruto[coluna_valor].reindex(preds.index, method=None)
    
    df_series = pd.DataFrame({
        "timestamp": preds.index,
        "valor_ft": valor_real.values,
        "previsao": preds.values,
        "anomalia_detectada": 0,
        "anomalia_persistente": 0
    })

    df_series = df_series[
        (df_series["timestamp"] >= data_ref) &
        (df_series["timestamp"] <= data_limite)
    ].copy().reset_index(drop=True)

    # 6. Verificar vazamento ativo
    vazamento_ativo = buscar_vazamento_ativo(estacao)

    if vazamento_ativo:
        ts_inicio_vazamento = pd.to_datetime(vazamento_ativo['timestamp'])
        mask_ativo = df_series['timestamp'] >= ts_inicio_vazamento
        df_series.loc[mask_ativo, 'anomalia_persistente'] = 1
        df_series.loc[mask_ativo, 'anomalia_detectada'] = 1
        print(f"ℹ️ Vazamento ativo desde {ts_inicio_vazamento} em {estacao}")

        # Tentar desativar se resolvido
        if desativar_vazamento_se_resolvido(estacao, df_series, LIMIAR_ABS, LIMIAR_PCT):
            df_series['anomalia_persistente'] = 0
            df_series['anomalia_detectada'] = 0
            vazamento_ativo = None

    # 7. Detectar NOVO vazamento (só se não há ativo)
    if not vazamento_ativo:
        # Evitar divisão por zero
        mask_valida = df_series['previsao'].notna() & (df_series['previsao'] > 1e-6)
        if mask_valida.any():
            erro_abs = df_series.loc[mask_valida, 'valor_ft'] - df_series.loc[mask_valida, 'previsao']
            erro_pct = erro_abs / df_series.loc[mask_valida, 'previsao']
            
            acima = (erro_abs > LIMIAR_ABS) | (erro_pct > LIMIAR_PCT)
            
            # Aplicar resultado de volta ao df_series completo
            df_series.loc[mask_valida, 'vazamento_candidato'] = acima
            df_series['vazamento_candidato'] = df_series['vazamento_candidato'].fillna(False).astype(bool)
            
            # Verificar persistência
            sequencia_4 = (
                df_series['vazamento_candidato']
                .astype(int)
                .rolling(window=4, min_periods=4)
                .sum() == 4
            )
            inicio_vazamento = sequencia_4.shift(-3).fillna(False)

            if inicio_vazamento.any():
                idx_inicio = inicio_vazamento.idxmax()
                ts_inicio = df_series.loc[idx_inicio, 'timestamp']
                valor_inicio = df_series.loc[idx_inicio, 'valor_ft']

                df_series.loc[df_series.index >= idx_inicio, 'anomalia_persistente'] = 1
                df_series.loc[df_series.index >= idx_inicio, 'anomalia_detectada'] = 1
                print(f"🚨 Novo vazamento detectado! Início: {ts_inicio} ({estacao})")
                salvar_inicio_vazamento(estacao, ts_inicio, valor_inicio, df_series.loc[idx_inicio, 'previsao'])
            else:
                print(f"✅ Sem novo vazamento em {estacao}")
        else:
            print(f"⚠️ Sem dados válidos para detecção em {estacao}")
    print(df_series)
    # 8. Salvar série temporal (descomente quando quiser)
    salvar_serie_temporal(estacao=estacao, df_series=df_series)
print(f"\n⏳ Tempo total: {time.time() - t0:.2f} segundos")
