"""
train.py
--------
Entrena el regresor LightGBM para predecir la temperatura óptima de
crecimiento (OGT) de proteínas a partir de sus 428 features bioquímicas.

Uso:
    python src/train.py

El script guarda dos artefactos en models/:
    - regresor_ogt.txt   : modelo LightGBM para predicción de temperatura
    - clasificador_ogt.txt : clasificador Psicrófilo/Mesófilo/Termófilo
"""

import gc
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.metrics import classification_report
import os

from features import extraer_features_por_lotes, NOMBRES_FEATURES

# ── Configuración ─────────────────────────────────────────────────────────

RUTA_X_TRAIN = os.getenv('RUTA_X_TRAIN', 'data/X_train.npy')
RUTA_Y_TRAIN = os.getenv('RUTA_Y_TRAIN', 'data/y_train.npy')
RUTA_MODELOS = os.getenv('RUTA_MODELOS', 'models/')

N_MUESTRAS  = 500_000   # Ajustar según RAM disponible
TAMANO_LOTE = 50_000
SEED        = 42

os.makedirs(RUTA_MODELOS, exist_ok=True)


# ── Funciones auxiliares ──────────────────────────────────────────────────

def calcular_pesos(y: np.ndarray, ratio_max: float = 6.0) -> np.ndarray:
    """
    Pesos de entrenamiento: 1x para mesófilos, 3x para termófilos
    moderados, 6x para termófilos extremos.
    """
    w = np.ones(len(y), dtype=np.float32)
    w[y > 65] = ratio_max
    w[(y > 45) & (y <= 65)] = ratio_max / 2
    return w


def categorizar(temp: float) -> str:
    if temp < 20:    return '1. Psicrófilo (<20°C)'
    elif temp <= 45: return '2. Mesófilo (20-45°C)'
    else:            return '3. Termófilo (>45°C)'


def evaluar_subgrupos(y_real: np.ndarray, y_pred: np.ndarray) -> None:
    grupos = {
        'Psicrófilo (<20°C)':      y_real < 20,
        'Mesófilo (20-45°C)':      (y_real >= 20) & (y_real <= 45),
        'Termófilo mod (45-65°C)': (y_real > 45)  & (y_real <= 65),
        'Termófilo ext (>65°C)':   y_real > 65,
    }
    print(f"  {'Grupo':<28} {'N':>8} {'MAE':>7} {'R²':>8} {'Sesgo':>8}")
    print(f"  {'-'*63}")
    for nombre, mask in grupos.items():
        if mask.sum() < 10:
            continue
        yr, yp = y_real[mask], y_pred[mask]
        r2_g   = r2_score(yr, yp) if yr.std() > 0 else float('nan')
        sesgo  = (yp - yr).mean()
        print(f"  {nombre:<28} {mask.sum():>8,} "
              f"{mean_absolute_error(yr, yp):>7.2f} "
              f"{r2_g:>8.3f} {sesgo:>+8.2f}")


# ── Pipeline principal ────────────────────────────────────────────────────

def main():
    # 1. Muestreo
    print('1. Cargando etiquetas y muestreando...')
    y_full = np.load(RUTA_Y_TRAIN)
    np.random.seed(SEED)
    idx = np.random.choice(len(y_full), size=N_MUESTRAS, replace=False)
    # ⚠️  NO ordenar idx — mantiene alineamiento X↔y
    y_sample = y_full[idx].copy()
    print(f'   {N_MUESTRAS:,} proteínas muestreadas')

    # 2. Extracción de features
    print('\n2. Extrayendo 428 features bioquímicas...')
    X = extraer_features_por_lotes(RUTA_X_TRAIN, idx,
                                    tamano_lote=TAMANO_LOTE)
    print(f'   X shape: {X.shape}  |  RAM: {X.nbytes/1e6:.0f} MB')

    # 3. Verificación de alineamiento X↔y
    idx_frio  = np.where(y_sample < 15)[0][:5]
    idx_calor = np.where(y_sample > 80)[0][:5]
    gravy_diff = abs(X[idx_calor, 21].mean() - X[idx_frio, 21].mean())
    assert gravy_diff > 0.05, (
        'GRAVY de proteínas frías y calientes son idénticos — '
        'posible desalineamiento X↔y. Revisar el pipeline.'
    )
    print(f'   Alineamiento verificado (ΔGRAVY frío/caliente = {gravy_diff:.3f})')

    # 4. Split estratificado
    bins = np.digitize(y_sample, bins=[20, 45, 65, 80])
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y_sample, test_size=0.2, random_state=SEED, stratify=bins
    )
    w = calcular_pesos(y_tr)
    print(f'\n3. Split: {len(X_tr):,} entrenamiento | {len(X_val):,} validación')

    # 5. Entrenamiento del regresor
    print('\n4. Entrenando regresor LightGBM...')
    params_reg = {
        'objective':         'regression',
        'metric':            ['mae', 'rmse'],
        'num_leaves':        127,
        'learning_rate':     0.05,
        'min_child_samples': 30,
        'subsample':         0.8,
        'subsample_freq':    1,
        'colsample_bytree':  0.4,
        'reg_alpha':         0.1,
        'reg_lambda':        0.5,
        'max_bin':           127,
        'num_threads':       8,
        'seed':              SEED,
        'verbose':           -1,
    }
    dtrain = lgb.Dataset(X_tr,  label=y_tr,  weight=w,
                         feature_name=NOMBRES_FEATURES, free_raw_data=False)
    dval   = lgb.Dataset(X_val, label=y_val,
                         feature_name=NOMBRES_FEATURES,
                         reference=dtrain, free_raw_data=False)
    dtrain.construct()

    regresor = lgb.train(
        params_reg, dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        valid_names=['val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=80, verbose=True),
            lgb.log_evaluation(period=100),
        ]
    )
    del dtrain, dval
    gc.collect()

    # 6. Evaluación del regresor
    pred_val = regresor.predict(X_val, num_iteration=regresor.best_iteration)
    mae  = mean_absolute_error(y_val, pred_val)
    rmse = np.sqrt(mean_squared_error(y_val, pred_val))
    r2   = r2_score(y_val, pred_val)

    print('\n' + '=' * 55)
    print('REGRESOR — VALIDACIÓN')
    print('=' * 55)
    print(f'   MAE:  {mae:.2f} °C')
    print(f'   RMSE: {rmse:.2f} °C')
    print(f'   R²:   {r2:.4f}')
    evaluar_subgrupos(y_val, pred_val)

    ruta_reg = os.path.join(RUTA_MODELOS, 'regresor_ogt.txt')
    regresor.save_model(ruta_reg)
    print(f'\n   Guardado: {ruta_reg}')

    # 7. Entrenamiento del clasificador
    print('\n5. Entrenando clasificador de screening...')
    y_tr_cat  = np.vectorize(categorizar)(y_tr)
    y_val_cat = np.vectorize(categorizar)(y_val)

    clasificador = lgb.LGBMClassifier(
        n_estimators=300,
        num_leaves=63,
        learning_rate=0.05,
        class_weight='balanced',
        subsample=0.8,
        colsample_bytree=0.4,
        n_jobs=-1,
        random_state=SEED,
        verbose=-1,
    )
    clasificador.fit(X_tr, y_tr_cat)

    pred_val_cat = clasificador.predict(X_val)
    print('\nCLASIFICADOR — VALIDACIÓN')
    print(classification_report(y_val_cat, pred_val_cat,
          target_names=['Psicrófilo', 'Mesófilo', 'Termófilo']))

    ruta_clas = os.path.join(RUTA_MODELOS, 'clasificador_ogt.txt')
    clasificador.booster_.save_model(ruta_clas)
    print(f'   Guardado: {ruta_clas}')


if __name__ == '__main__':
    main()
