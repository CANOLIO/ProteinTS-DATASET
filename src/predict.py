"""
predict.py
----------
Predice la temperatura óptima de crecimiento (OGT) y la categoría funcional
de nuevas proteínas a partir de sus secuencias codificadas.

Uso como script:
    python src/predict.py --x nueva_proteina.npy --modelo models/regresor_ogt.txt

Uso como módulo:
    from predict import predecir
    temperaturas = predecir(X_nuevas, modelo_path='models/regresor_ogt.txt')
"""

import argparse
import numpy as np
import lightgbm as lgb

from features import extraer_features


def predecir(X_seqs: np.ndarray, modelo_path: str) -> np.ndarray:
    """
    Predice OGT (°C) para un array de secuencias codificadas.

    Parámetros
    ----------
    X_seqs : np.ndarray, shape (N, L)
        Secuencias codificadas como enteros 0-20.
    modelo_path : str
        Ruta al archivo .txt del modelo LightGBM guardado con save_model().

    Retorna
    -------
    np.ndarray, shape (N,), dtype float32
        Temperaturas predichas en grados Celsius.
    """
    modelo   = lgb.Booster(model_file=modelo_path)
    features = extraer_features(X_seqs)
    return modelo.predict(features).astype(np.float32)


def clasificar(X_seqs: np.ndarray, modelo_path: str) -> np.ndarray:
    """
    Clasifica proteínas en Psicrófilo / Mesófilo / Termófilo.

    Retorna array de strings con la categoría predicha.
    """
    modelo   = lgb.Booster(model_file=modelo_path)
    features = extraer_features(X_seqs)
    return np.array(modelo.predict(features))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predice OGT de proteínas a partir de secuencias codificadas'
    )
    parser.add_argument('--x',       required=True,
                        help='Ruta al .npy de secuencias (shape NxL, enteros 0-20)')
    parser.add_argument('--modelo',  default='models/regresor_ogt.txt',
                        help='Ruta al modelo LightGBM (.txt)')
    parser.add_argument('--salida',  default=None,
                        help='Ruta para guardar predicciones como .npy (opcional)')
    args = parser.parse_args()

    print(f'Cargando secuencias desde {args.x}...')
    X = np.load(args.x)
    print(f'  Shape: {X.shape}')

    print('Extrayendo features y prediciendo...')
    temps = predecir(X, args.modelo)

    print(f'\nPredicciones (primeras 10): {temps[:10].round(1)}')
    print(f'Rango: {temps.min():.1f} - {temps.max():.1f} °C')
    print(f'Media: {temps.mean():.2f} °C')

    if args.salida:
        np.save(args.salida, temps)
        print(f'Guardado en: {args.salida}')
