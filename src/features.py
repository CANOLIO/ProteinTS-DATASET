"""
features.py
-----------
Extracción de features bioquímicas a partir de secuencias de proteínas
codificadas como arrays de enteros (0=padding, 1-20=aminoácidos).

Features extraídas (428 total):
  - 20 frecuencias relativas de aminoácidos (monopéptidos)
  -  8 propiedades bioquímicas: GRAVY, aromaticidad, log-longitud,
       % Cys, carga positiva, carga negativa, balance de cargas, % Pro
  - 400 frecuencias relativas de dipéptidos (pares consecutivos)

Codificación de aminoácidos:
  1=A  2=C  3=D  4=E  5=F  6=G  7=H  8=I  9=K  10=L
  11=M 12=N 13=P 14=Q 15=R 16=S 17=T 18=V 19=W 20=Y
"""

import numpy as np

# ── Constantes ────────────────────────────────────────────────────────────

# Índices de hidrofobicidad de Kyte-Doolittle
# Referencia: Kyte & Doolittle (1982) J Mol Biol 157:105-132
KD = np.array([
     1.8,  2.5, -3.5, -3.5,  2.8, -0.4, -3.2,  4.5, -3.9,  3.8,
     1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7,  4.2, -0.9, -1.3
], dtype=np.float32)

AA_LETRAS = ['A','C','D','E','F','G','H','I','K','L',
             'M','N','P','Q','R','S','T','V','W','Y']

NOMBRES_MONO = (
    [f'pct_{aa}' for aa in AA_LETRAS] +
    ['log_longitud', 'GRAVY', 'aromaticidad', 'pct_Cys',
     'carga_pos', 'carga_neg', 'balance_cargas', 'pct_Pro']
)  # 28 features

NOMBRES_DI = [f'di_{a}{b}' for a in AA_LETRAS for b in AA_LETRAS]  # 400 features

NOMBRES_FEATURES = NOMBRES_MONO + NOMBRES_DI  # 428 total
N_FEATURES = len(NOMBRES_FEATURES)


# ── Función principal ─────────────────────────────────────────────────────

def extraer_features(X_seqs: np.ndarray) -> np.ndarray:
    """
    Extrae 428 features bioquímicas de un array de secuencias codificadas.

    Parámetros
    ----------
    X_seqs : np.ndarray, shape (N, L)
        Matriz de secuencias. Cada fila es una proteína codificada como
        enteros 0-20, donde 0 es padding y 1-20 son los aminoácidos.

    Retorna
    -------
    np.ndarray, shape (N, 428), dtype float32
        Matriz de features. Ver NOMBRES_FEATURES para el detalle de cada columna.

    Notas
    -----
    - El padding (0) se ignora automáticamente, incluyendo ceros embebidos.
    - Los dipéptidos que cruzan una posición de padding se excluyen.
    - La función es segura ante arrays de dtype object, uint8 o int64.
    """
    N = X_seqs.shape[0]
    F = np.zeros((N, N_FEATURES), dtype=np.float32)

    for i in range(N):
        # Filtrar padding y forzar dtype seguro para bincount
        aa = X_seqs[i]
        aa = aa[aa > 0].astype(np.int64)
        L = len(aa)
        if L == 0:
            continue

        # ── Monopéptidos (columnas 0-19) ──────────────────────────────────
        c = np.bincount(aa, minlength=21)[1:21].astype(np.float32)
        F[i, :20] = c / L * 100

        # ── Features bioquímicas (columnas 20-27) ─────────────────────────
        F[i, 20] = np.log1p(L)                          # longitud (log)
        F[i, 21] = np.dot(c, KD) / L                    # GRAVY score
        F[i, 22] = (c[4] + c[19] + c[18]) / L * 100    # aromaticidad F+Y+W
        F[i, 23] = c[1] / L * 100                       # % Cisteína
        F[i, 24] = (c[8] + c[14]) / L * 100             # carga+ (K+R)
        F[i, 25] = (c[2] + c[3]) / L * 100              # carga- (D+E)
        F[i, 26] = F[i, 24] - F[i, 25]                  # balance de cargas
        F[i, 27] = c[12] / L * 100                      # % Prolina

        # ── Dipéptidos (columnas 28-427) ──────────────────────────────────
        # Índice de par (a,b) = (a-1)*20 + (b-1), donde a,b ∈ [1,20]
        # Se filtran pares que crucen posiciones de padding
        if L >= 2:
            a_left  = aa[:-1]
            a_right = aa[1:]
            validos = (
                (a_left  >= 1) & (a_left  <= 20) &
                (a_right >= 1) & (a_right <= 20)
            )
            a_left  = a_left[validos]
            a_right = a_right[validos]
            n_pares = len(a_left)
            if n_pares > 0:
                pares = (a_left - 1) * 20 + (a_right - 1)
                conteos_di = np.bincount(pares, minlength=400).astype(np.float32)
                F[i, 28:428] = conteos_di / n_pares * 100

    return F


def extraer_features_por_lotes(ruta_npy: str,
                                indices: np.ndarray,
                                tamano_lote: int = 50_000,
                                verbose: bool = True) -> np.ndarray:
    """
    Extrae features de un subconjunto de secuencias cargado desde disco
    por lotes, para no saturar la RAM.

    Parámetros
    ----------
    ruta_npy : str
        Ruta al archivo .npy de secuencias (shape NxL).
    indices : np.ndarray
        Índices de las proteínas a procesar. NO deben estar ordenados
        si se quiere mantener el alineamiento con un array y externo.
    tamano_lote : int
        Número de secuencias a cargar por lote. Por defecto 50_000
        (~100 MB por lote, seguro en 6GB de RAM).
    verbose : bool
        Si True, imprime progreso por lote.

    Retorna
    -------
    np.ndarray, shape (len(indices), 428), dtype float32
    """
    import gc
    X_mmap = np.load(ruta_npy, mmap_mode='r')
    N = len(indices)
    n_lotes = int(np.ceil(N / tamano_lote))
    bloques = []

    for k in range(n_lotes):
        i0 = k * tamano_lote
        i1 = min(i0 + tamano_lote, N)
        X_lote = np.array(X_mmap[indices[i0:i1]])
        bloques.append(extraer_features(X_lote))
        del X_lote
        if verbose and ((k + 1) % 3 == 0 or k == n_lotes - 1):
            print(f'  Lote {k+1}/{n_lotes} — {i1:,}/{N:,} proteínas')
        gc.collect()

    return np.vstack(bloques)
