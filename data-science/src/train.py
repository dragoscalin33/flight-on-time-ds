import pandas as pd
import numpy as np
import joblib
import holidays
import os
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, accuracy_score, classification_report
from sklearn.base import BaseEstimator, TransformerMixin

# --- 1. DEFINIÇÃO DA CLASSE SAFE ENCODER (CRÍTICO) ---
class SafeLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.classes_ = {}
        self.unknown_token = -1

    def fit(self, y):
        unique_labels = pd.Series(y).unique()
        self.classes_ = {label: idx for idx, label in enumerate(unique_labels)}
        return self

    def transform(self, y):
        # Converte para string para evitar erros de tipo e mapeia
        return pd.Series(y).apply(lambda x: self.classes_.get(str(x), self.unknown_token))

# --- FUNÇÕES AUXILIARES ---
def haversine_distance(lat1, lon1, lat2, lon2):
    r = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    return r * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

# --- CONFIGURAÇÃO ---
print(" Iniciando treinamento V4.0 (Weather-Aware)...")
current_dir = os.path.dirname(__file__)

# [MUDANÇA V4] Apontamos para o dataset enriquecido com clima
data_path = os.path.join(current_dir, '../data/BrFlights_Enriched_v4.csv') 
# [MUDANÇA V4] Nome do arquivo final atualizado
model_path = os.path.join(current_dir, 'flight_classifier_v4.joblib')

# 2. CARGA DE DADOS
try:
    # low_memory=False ajuda com datasets grandes
    df = pd.read_csv(data_path, low_memory=False)
    print(f"✅ Registros carregados: {len(df):,}")
except FileNotFoundError:
    print(f" Erro: Arquivo não encontrado em {data_path}")
    print("   -> Execute o Notebook 1 para gerar o dataset enriquecido.")
    exit()

# 3. LIMPEZA E ENGENHARIA
print("🛠️ Criando features e aplicando limpeza...")

# A. Distância (Garantindo numérico)
for col in ['LatOrig', 'LongOrig', 'LatDest', 'LongDest']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['distancia_km'] = haversine_distance(df['LatOrig'], df['LongOrig'], df['LatDest'], df['LongDest'])

# B. Datas
cols_datas = ['Partida.Prevista', 'Partida.Real', 'Chegada.Real']
for col in cols_datas:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# C. Filtragem Básica
# [MUDANÇA V4] Remover cancelados e nulos críticos
if 'Situacao.Voo' in df.columns:
    df_clean = df[df['Situacao.Voo'] == 'Realizado'].copy()
else:
    df_clean = df.copy() # Assumindo que o CSV v4 já vem meio limpo

df_clean = df_clean.dropna(subset=cols_datas + ['distancia_km'])

# D. Features Calculadas (Target)
df_clean['delay_minutes'] = (df_clean['Partida.Real'] - df_clean['Partida.Prevista']).dt.total_seconds() / 60
df_clean['duration_minutes'] = (df_clean['Chegada.Real'] - df_clean['Partida.Real']).dt.total_seconds() / 60

# E. Outliers
mask_clean = (df_clean['duration_minutes'] > 0) & (df_clean['delay_minutes'] > -60) & (df_clean['delay_minutes'] < 1440)
df_clean = df_clean[mask_clean].copy()

# F. Target (> 15 min)
df_clean['target'] = np.where(df_clean['delay_minutes'] > 15, 1, 0)

# G. Variáveis Temporais e Climáticas
print(" Processando Clima e Calendário...")
br_holidays = holidays.Brazil()
df_clean['is_holiday'] = df_clean['Partida.Prevista'].dt.date.apply(lambda x: 1 if x in br_holidays else 0)

df_clean['hora'] = df_clean['Partida.Prevista'].dt.hour
df_clean['dia_semana'] = df_clean['Partida.Prevista'].dt.dayofweek
df_clean['mes'] = df_clean['Partida.Prevista'].dt.month

# [MUDANÇA V4] Garantir que colunas de clima existem e são float
for col in ['precipitation', 'wind_speed']:
    if col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
    else:
        print(f" Aviso: Coluna {col} não encontrada. Preenchendo com 0.")
        df_clean[col] = 0.0

# Renomear
df_clean.rename(columns={'Companhia.Aerea': 'companhia', 'Aeroporto.Origem': 'origem', 'Aeroporto.Destino': 'destino'}, inplace=True)

# 4. SPLIT E ENCODING
print(" Realizando Split e Encoding Seguro...")

# [MUDANÇA V4] Adicionamos precipitation e wind_speed na lista base
cols_base = [
    'companhia', 'origem', 'destino', 
    'distancia_km', 'hora', 'dia_semana', 'mes', 'is_holiday',
    'precipitation', 'wind_speed'
]

X = df_clean[cols_base]
y = df_clean['target']

# Split Estratificado
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Encoding Seguro
encoders = {}
cat_features = ['companhia', 'origem', 'destino']

for col in cat_features:
    le = SafeLabelEncoder()
    # Convertemos para string para garantir robustez
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)
    
    # Fit no Treino, Transform no Teste
    X_train[f'{col}_encoded'] = le.fit(X_train[col]).transform(X_train[col])
    X_test[f'{col}_encoded'] = le.transform(X_test[col])
    encoders[col] = le

# [MUDANÇA V4] Lista final de features numéricas para o modelo
features_finais = [
    'companhia_encoded', 'origem_encoded', 'destino_encoded', 
    'distancia_km', 'hora', 'dia_semana', 'mes', 'is_holiday',
    'precipitation', 'wind_speed' # Novas!
]

# 5. TREINAMENTO
print(f" Treinando CatBoost Classifier V4...")
model = CatBoostClassifier(
    iterations=300,            # [MUDANÇA V4] Aumentado para 300 para capturar padrões complexos
    learning_rate=0.1,
    depth=6,
    auto_class_weights='Balanced',
    random_seed=42,
    verbose=50,
    allow_writing_files=False
)

# Treina apenas com as colunas numéricas finais
model.fit(X_train[features_finais], y_train)

# 6. VALIDAÇÃO E MÉTRICAS
THRESHOLD = 0.40 # [MUDANÇA V4] Threshold definido na análise de negócio
probs = model.predict_proba(X_test[features_finais])[:, 1]
preds = (probs >= THRESHOLD).astype(int)

recall = recall_score(y_test, preds)
acc = accuracy_score(y_test, preds)

print("-" * 40)
print(f"🎯 Resultado Final (Threshold {THRESHOLD}):")
print(f"   -> Recall:   {recall:.2%}")
print(f"   -> Accuracy: {acc:.2%}")
print("-" * 40)

# 7. EXPORTAR ARTEFATO COMPLETO
print(" Salvando artefatos de produção...")

# Re-treinar com TODO o dataset (Opcional, mas recomendado para produção)
# Aqui, por simplicidade no script, salvaremos o modelo treinado no X_train, 
# mas em produção real idealmente concatenaríamos X_train + X_test.

artifact = {
    'model': model,
    'encoders': encoders,
    'features': features_finais,
    'metadata': {
        'autor': 'Time Data Science',
        'versao': '4.0.0-WeatherAware',
        'tecnologia': 'CatBoost + OpenMeteo',
        'threshold': THRESHOLD,
        'recall_esperado': f"{recall:.2%}"
    }
}

joblib.dump(artifact, model_path)
print(f"✅ Arquivo gerado com sucesso: {model_path}")
print(" Pronto para deploy!")