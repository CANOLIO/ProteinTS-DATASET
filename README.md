# Predicción de Termoestabilidad de Proteínas

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?logo=pandas)

Modelo de machine learning para predecir la **temperatura óptima de crecimiento (OGT)** del organismo de origen a partir de la composición de aminoácidos de sus proteínas, utilizando features bioquímicas derivadas de la secuencia.

---

## Motivación

La termoestabilidad de las proteínas es una propiedad crítica en biotecnología industrial: las enzimas de organismos termófilos (bacterias que viven a >45°C) son más resistentes a la desnaturalización, lo que las hace valiosas en procesos industriales como la producción de biocombustibles, detergentes y fármacos.

Este proyecto explora hasta dónde puede llegar un modelo clásico de machine learning (sin redes neuronales) para predecir esta propiedad usando únicamente la secuencia de aminoácidos, evaluado sobre un conjunto de datos de ~7.7 millones de proteínas.

---

## Dataset

**Fuente:** [Kaggle — Enzyme Thermostability](https://www.kaggle.com/)

| Archivo | Forma | Descripción |
|---------|-------|-------------|
| `X_train.npy` | (6,149,359 × 650) | Secuencias codificadas (enteros 0-20) |
| `y_train.npy` | (6,149,359,) | OGT del organismo en °C |
| `X_test.npy`  | (1,537,340 × 650) | Secuencias de evaluación |
| `y_test.npy`  | (1,537,340,) | OGT real para evaluación |

Cada secuencia está codificada como un vector de longitud 650 con padding de ceros al final (y en algunos casos embebido). La distribución de temperaturas es fuertemente asimétrica:

| Categoría | Rango | % del dataset |
|-----------|-------|---------------|
| Psicrófilo | < 20°C | ~2.5% |
| Mesófilo | 20 – 45°C | ~83% |
| Termófilo moderado | 45 – 65°C | ~10% |
| Termófilo extremo | > 65°C | ~4.5% |

---

## Feature Engineering

Se extraen **428 features** por proteína, organizadas en tres grupos:

### 1. Frecuencias de aminoácidos — 20 features
Porcentaje de cada uno de los 20 aminoácidos canónicos en la secuencia. Base clásica para predicción de propiedades de proteínas.

### 2. Propiedades bioquímicas derivadas — 8 features

| Feature | Descripción | Relevancia |
|---------|-------------|------------|
| `GRAVY` | Índice de hidrofobicidad (Kyte-Doolittle) | Proteínas termófilas tienden a tener GRAVY más positivo |
| `aromaticidad` | % Phe + Tyr + Trp | Los anillos aromáticos forman interacciones π-π estabilizadoras |
| `pct_Cys` | % Cisteína | Los puentes disulfuro confieren rigidez estructural |
| `carga_pos` | % Lys + Arg | |
| `carga_neg` | % Asp + Glu | |
| `balance_cargas` | carga_pos − carga_neg | Correlaciona con punto isoeléctrico estimado |
| `pct_Pro` | % Prolina | Las prolinas rigidizan la cadena peptídica |
| `log_longitud` | log(1 + longitud real) | Proteínas termoestables tienden a ser más compactas |

### 3. Dipéptidos — 400 features
Frecuencia relativa de cada uno de los 20×20=400 pares de aminoácidos consecutivos posibles. Capturan patrones de orden local en la secuencia que los monopéptidos no pueden ver y reflejan tendencias de estructura secundaria (hélices α, láminas β).

---

## Modelo

**Algoritmo:** LightGBM (Gradient Boosting con histogramas comprimidos)

Se entrenaron dos modelos sobre una muestra de 500,000 proteínas con distribución natural:

### Regresor de OGT
Predice la temperatura óptima de crecimiento en °C.

```
Hiperparámetros principales:
  num_leaves:        127
  learning_rate:     0.05
  colsample_bytree:  0.4   (40% de 428 features por árbol)
  subsample:         0.8
  early_stopping:    80 rondas sin mejora en validación
```

El desbalance de clases se maneja con `sample_weight`:
- Termófilos extremos (>65°C): peso 6×
- Termófilos moderados (45-65°C): peso 3×
- Resto: peso 1×

### Clasificador de Screening
Clasifica proteínas en Psicrófilo / Mesófilo / Termófilo usando `class_weight='balanced'`.

---

## Resultados

### Regresor — Test Set real (1,537,340 proteínas)

| Métrica | Valor |
|---------|-------|
| **MAE** | **7.57 °C** |
| **RMSE** | **10.23 °C** |
| **R²** | **0.350** |

![Test set real](images/evfinal.png)

### Clasificador — Test Set real (1,537,340 proteínas)

| Subgrupo | N | MAE | R² | Sesgo |
|----------|---|-----|----|-------|
| Psicrófilo (<20°C) | 38,546 | 18.65°C | −29.3 | +18.65°C |
| Mesófilo (20-45°C) | 1,282,663 | 6.17°C | −1.4 | +3.69°C |
| Termófilo mod. (45-65°C) | 160,398 | 12.15°C | −6.1 | −10.90°C |
| Termófilo ext. (>65°C) | 55,733 | 18.99°C | −5.9 | −18.12°C |

![Clasificador](images/clasificador.png)

> **Contexto:** Un predictor constante (siempre la media, ~35°C) tendría MAE ≈ 12°C y R²=0. El modelo reduce el MAE un 37% respecto a ese baseline. El R² bajo es esperado para este tipo de datos: la OGT es una propiedad del *organismo*, no de la proteína individual, y dos proteínas del mismo organismo tienen la misma temperatura pero composiciones muy distintas. La literatura reporta R² ≈ 0.30-0.45 como techo teórico para modelos basados solo en composición de aminoácidos (Zeldovich et al., 2007; Engqvist, 2018).

---

## Limitaciones y trabajo futuro

El sesgo sistemático visible en los extremos (psicrófilos sobreestimados, termófilos subestimados) indica que la composición de aminoácidos tiene señal moderada pero insuficiente para separar bien los grupos extremos. Para superar R² ≈ 0.45 se necesitaría:

- **Embeddings de secuencia completa** (ESM-2, ProtBERT) que capturan información de estructura 3D implícita.
- **Trigramas selectivos**: los ~50 trigramas más correlacionados con temperatura, sin llegar a los 8000 posibles.
- **Autocorrelación de hidrofobicidad** a lags 3-4 (firma de hélices α) y lag 2 (láminas β).
- **Momento hidrofóbico de Eisenberg**: cuantifica asimetría de distribución de hidrofobicidad, diferencia bien hélices anfipáticas de termófilos.

---

## Estructura del repositorio

```
protein-thermostability/
│
├── notebooks/
│   └── thermostability_pipeline.ipynb   # Análisis exploratorio y desarrollo del pipeline
│
├── src/
│   ├── features.py     # Extracción de 428 features bioquímicas
│   ├── train.py        # Entrenamiento del regresor y clasificador
│   ├── evaluate.py     # Evaluación en test set por lotes
│   └── predict.py      # Inferencia sobre nuevas secuencias
│
├── models/             # Modelos entrenados (no incluidos en el repo)
│   ├── regresor_ogt.txt
│   └── clasificador_ogt.txt
│
├── data/               # Datos del dataset (no incluidos en el repo)
│   ├── X_train.npy
│   ├── y_train.npy
│   ├── X_test.npy
│   └── y_test.npy
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Instalación y uso

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/protein-thermostability.git
cd protein-thermostability

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Colocar los datos en data/
# (descargar desde Kaggle y mover los .npy a la carpeta data/)

# 4. Entrenar los modelos
python src/train.py

# 5. Evaluar en el test set
python src/evaluate.py

# 6. Predecir sobre nuevas secuencias
python src/predict.py --x data/X_test.npy --modelo models/regresor_ogt.txt
```

Para usar las rutas de datos personalizadas:
```bash
RUTA_X_TRAIN=/ruta/X_train.npy RUTA_Y_TRAIN=/ruta/y_train.npy python src/train.py
```

---

## Referencias

- Zeldovich, K.B., et al. (2007). *Protein and DNA sequence determinants of thermophilic adaptation*. PNAS, 104(42), 16516-16521.
- Engqvist, M.K.M. (2018). *Correlating enzyme annotations with a large set of microbial growth temperatures reveals metabolic adaptations to growth at diverse temperatures*. BMC Microbiology, 18, 177.
- Kyte, J. & Doolittle, R.F. (1982). *A simple method for displaying the hydropathic character of a protein*. Journal of Molecular Biology, 157(1), 105-132.
- Ke, G., et al. (2017). *LightGBM: A highly efficient gradient boosting decision tree*. NeurIPS.

---

## Autor

**Fabián** — Bioquímico  
Proyecto desarrollado con mucha pasión como ejercicio de machine learning aplicado a bioinformática estructural. 
