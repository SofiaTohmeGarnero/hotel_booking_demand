# 🏨 Proyecto: Predicción de Cancelaciones en Reservas de Hotel

Este repositorio contiene el código, los datos y los experimentos para analizar y predecir cancelaciones de reservas hoteleras. Trabajaremos sobre el dataset `hotel_bookings.csv` y construiremos flujos de aprendizaje automático usando pandas, scikit-learn y Matplotlib/Seaborn.

## 📁 Estructura del Repositorio

Asegúrate de mantener esta estructura. Ten en cuenta que la carpeta `data/` está ignorada en el control de versiones (Git) por buenas prácticas y seguridad.

*   `data/` 
    *   `raw/`: Dataset original descargado de Kaggle.
    *   `time-based-splits/`: Subconjuntos iniciales (`train_before_eda.csv`, `test.csv`) particionados cronológicamente.
    *   `processed/`: Datasets limpios y con *feature engineering* aplicado (`train_clean.csv`, `test_clean.csv`).
    *   `model_input/`
        *   `static/`: Matrices de características y etiquetas (X, y) para entrenamiento con validación estática (Train 70%, Val 15%, Test 15%).
        *   `cv/`: Matrices de características y etiquetas (X, y) preparadas para validación cruzada temporal (Train Full 85%, Test 15%).
*   `docs/`: Documentación adicional del proyecto, referencias y notas.
*   `models/`: Modelos de Machine Learning entrenados y guardados para su posterior uso o despliegue.
*   `notebooks/`: Jupyter Notebooks para el Análisis Exploratorio de Datos (EDA) y experimentación de modelos.
*   `scripts/`: Scripts automatizados de Python para la ingesta, preparación y procesamiento de datos.

## 🚀 Instrucciones de Configuración (Para el equipo)

Para garantizar la reproducibilidad matemática y que todos trabajemos exactamente con las mismas particiones de datos, sigue estos pasos en orden ejecutando los scripts:

**1. Descargar el dataset original**
Conecta con Kaggle y descarga el archivo `hotel_bookings.csv` guardándolo en `data/raw/`.
> `python scripts/01_download.py`

**2. Partición temporal inicial**
Elimina registros duplicados, crea la variable `booking_date` y realiza un corte cronológico (85% Train, 15% Test).
> `python scripts/02_time-based-splitting.py`

**3. Preprocesamiento y Feature Engineering**
Aplica las reglas de limpieza definidas en el EDA (manejo de nulos, outliers, data leakage) a los subconjuntos temporales.
> `python scripts/03_preprocessing.py`

**4. Preparar matrices para modelado (Elige una o ambas opciones)**
*   **Para enfoque de Split Estático:** Genera las matrices X e y dividiendo en Train, Validation y Test.
    > `python scripts/04a_prep_static_split.py`
*   **Para enfoque de Cross Validation:** Genera las matrices X e y para usar con `TimeSeriesSplit`.
    > `python scripts/04b_prep_cv_split.py`

**⚠️ Regla de Oro para el Análisis Exploratorio (EDA)**
Al crear Notebooks para EDA, **debes cargar única y exclusivamente** el archivo `data/time-based-splits/train_before_eda.csv` para evitar la fuga de información (data leakage).

---

## 📓 Buenas Prácticas con Jupyter Notebooks y Git

Los archivos `.ipynb` guardan tanto el código como las salidas (gráficos, tablas). Subir las salidas a GitHub genera conflictos difíciles de resolver y ensucia el historial. Por favor, adopten una de las siguientes prácticas antes de hacer un commit:

*   **La buena práctica básica:** Antes de guardar y hacer un commit, ve al menú superior de Jupyter y selecciona **"Kernel -> Restart & Clear Output"** (Reiniciar y limpiar salidas). Esto borra las tablas y gráficos guardados, subiendo solo tu código Python limpio.
*   **La alternativa avanzada:** Recomendada para la industria. Instalaremos una herramienta llamada `nbstripout`, que limpia automáticamente las salidas de los notebooks justo antes de que Git haga el commit, sin que tengas que acordarte de hacerlo manualmente. Para configurarlo, ejecuta en tu terminal:
    ```bash
    pip install nbstripout
    nbstripout --install
    ```
