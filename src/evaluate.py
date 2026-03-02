"""
evaluate.py
-----------
Evalúa el regresor y el clasificador entrenados sobre el test set completo
(~1.5M proteínas), procesando por lotes para no saturar la RAM.

Uso:
    python src/evaluate.py

Genera en models/:
    - resultados_test.txt : métricas globales y por subgrupo
"""

import gc
import os
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (mean_absolute_error, r2_score,
                              mean_squared_error, classification_report,
                              confusion_matrix)

from features import extraer_features, NOMBRES_FEATURES

# ── Configuración ─────────────────────────────────────────────────────────

RUTA_X_TEST   = os.getenv('RUTA_X_TEST',   'data/X_test.npy')
RUTA_Y_TEST   = os.getenv('RUTA_Y_TEST',   'data/y_test.npy')
RUTA_MODELOS  = os.getenv('RUTA_MODELOS',  'models/')
TAMANO_LOTE   = 50_000


def categorizar(temp):
    if temp < 20:    return '1. Psicrófilo (<20°C)'
    elif temp <= 45: return '2. Mesófilo (20-45°C)'
    else:            return '3. Termófilo (>45°C)'


def predecir_por_lotes(modelo, ruta_x: str, n_total: int,
                        clasificador=None) -> tuple:
    """
    Carga X_test por lotes, extrae features y predice.
    Retorna (pred_reg, pred_cls) donde pred_cls es None si no hay clasificador.
    """
    X_mmap   = np.load(ruta_x, mmap_mode='r')
    pred_reg = np.zeros(n_total, dtype=np.float32)
    pred_cls = [] if clasificador else None

    n_lotes = int(np.ceil(n_total / TAMANO_LOTE))
    for k in range(n_lotes):
        i0 = k * TAMANO_LOTE
        i1 = min(i0 + TAMANO_LOTE, n_total)

        X_lote  = np.array(X_mmap[i0:i1])
        feats   = extraer_features(X_lote)
        pred_reg[i0:i1] = modelo.predict(feats)

        if clasificador:
            pred_cls.extend(clasificador.predict(feats))

        del X_lote, feats
        if (k + 1) % 10 == 0 or k == n_lotes - 1:
            print(f'  Lote {k+1}/{n_lotes} ({i1:,}/{n_total:,})')
        gc.collect()

    return pred_reg, pred_cls


def evaluar_subgrupos(y_real, y_pred, titulo=''):
    grupos = {
        'Psicrófilo (<20°C)':      y_real < 20,
        'Mesófilo (20-45°C)':      (y_real >= 20) & (y_real <= 45),
        'Termófilo mod (45-65°C)': (y_real > 45)  & (y_real <= 65),
        'Termófilo ext (>65°C)':   y_real > 65,
    }
    lineas = [titulo, f"  {'Grupo':<28} {'N':>9} {'MAE':>7} {'R²':>8} {'Sesgo':>8}",
              f"  {'-'*63}"]
    for nombre, mask in grupos.items():
        if mask.sum() < 10:
            continue
        yr, yp  = y_real[mask], y_pred[mask]
        r2_g    = r2_score(yr, yp) if yr.std() > 0 else float('nan')
        sesgo   = (yp - yr).mean()
        lineas.append(
            f"  {nombre:<28} {mask.sum():>9,} "
            f"{mean_absolute_error(yr, yp):>7.2f} "
            f"{r2_g:>8.3f} {sesgo:>+8.2f}"
        )
    return '\n'.join(lineas)


def main():
    # Cargar etiquetas
    print('Cargando test set...')
    y_test  = np.load(RUTA_Y_TEST)
    N_TEST  = len(y_test)
    print(f'  {N_TEST:,} proteínas')

    # Cargar modelos
    ruta_reg  = os.path.join(RUTA_MODELOS, 'regresor_ogt.txt')
    ruta_clas = os.path.join(RUTA_MODELOS, 'clasificador_ogt.txt')

    print('\nCargando modelos...')
    regresor     = lgb.Booster(model_file=ruta_reg)
    clasificador = lgb.LGBMClassifier()
    clasificador = lgb.Booster(model_file=ruta_clas)

    # Predicciones por lotes
    print('\nGenerando predicciones por lotes...')
    pred_reg, pred_cls = predecir_por_lotes(
        regresor, RUTA_X_TEST, N_TEST, clasificador=clasificador
    )

    # ── Métricas del regresor ─────────────────────────────────────────────
    mae  = mean_absolute_error(y_test, pred_reg)
    rmse = np.sqrt(mean_squared_error(y_test, pred_reg))
    r2   = r2_score(y_test, pred_reg)

    reporte_reg = (
        '\n' + '=' * 60 + '\n'
        'REGRESOR — TEST SET REAL\n'
        + '=' * 60 + '\n'
        f'   MAE:  {mae:.2f} °C\n'
        f'   RMSE: {rmse:.2f} °C\n'
        f'   R²:   {r2:.4f}\n'
        + evaluar_subgrupos(y_test, pred_reg, '\nPor subgrupo:')
    )

    # ── Métricas del clasificador ─────────────────────────────────────────
    y_test_cat = np.vectorize(categorizar)(y_test)
    reporte_cls = (
        '\n\n' + '=' * 60 + '\n'
        'CLASIFICADOR DE SCREENING — TEST SET REAL\n'
        + '=' * 60 + '\n'
        + classification_report(y_test_cat, pred_cls,
              target_names=['Psicrófilo', 'Mesófilo', 'Termófilo'])
    )

    # Imprimir y guardar
    reporte_completo = reporte_reg + reporte_cls
    print(reporte_completo)

    ruta_out = os.path.join(RUTA_MODELOS, 'resultados_test.txt')
    with open(ruta_out, 'w', encoding='utf-8') as f:
        f.write(reporte_completo)
    print(f'\nResultados guardados en: {ruta_out}')


if __name__ == '__main__':
    main()
