# 🏨 Proyecto: Predicción de Cancelaciones en Reservas de Hotel

Este repositorio contiene el código, los datos y los experimentos para analizar y predecir cancelaciones de reservas hoteleras. Trabajaremos sobre el dataset `hotel_bookings.csv` y construiremos flujos de aprendizaje automático usando pandas, scikit-learn y Matplotlib/Seaborn.

## 📁 Estructura del Repositorio

Asegúrate de mantener esta estructura. Ten en cuenta que la carpeta `data/` está ignorada en el control de versiones (Git) por buenas prácticas y seguridad.

*   `data/` 
    *   `raw/`: Dataset original.
    *   `splits/`: Subconjuntos limpios y estratificados (`train_before_eda.csv`, `test.csv`).
*   `notebooks/`: Jupyter Notebooks para el Análisis Exploratorio de Datos (EDA) y experimentación de modelos.
*   `scripts/`: Scripts automatizados de Python para la ingesta y preparación de datos.

## 🚀 Instrucciones de Configuración (Para el equipo)

Para garantizar la reproducibilidad matemática y que todos trabajemos exactamente con las mismas particiones de datos, sigue estos pasos en orden:

**1. Abrir la terminal**
Ubícate en la carpeta raíz del proyecto desde tu línea de comandos.

**2. Descargar el dataset original**
Ejecuta el script de descarga. Esto obtendrá el archivo `hotel_bookings.csv` y lo guardará en `data/raw/`.
> `python scripts/01_download.py`

**3. Limpiar y particionar los datos**
Ejecuta el script de preparación. Este código eliminará los registros duplicados y creará los cortes estratificados de Entrenamiento, Validación y Prueba usando una semilla fija (`random_state=42`).
> `python scripts/02_prepare.py`

**4. Regla de Oro para el Análisis**
Al crear Notebooks para EDA o entrenamiento inicial, **debes cargar única y exclusivamente** el archivo `data/splits/train_before_eda.csv` para evitar la fuga de información (data leakage).

---

## 📓 Buenas Prácticas con Jupyter Notebooks y Git

Los archivos `.ipynb` guardan tanto el código como las salidas (gráficos, tablas). Subir las salidas a GitHub genera conflictos difíciles de resolver y ensucia el historial. Por favor, adopten una de las siguientes prácticas antes de hacer un commit:

*   **La buena práctica básica:** Antes de guardar y hacer un commit, ve al menú superior de Jupyter y selecciona **"Kernel -> Restart & Clear Output"** (Reiniciar y limpiar salidas). Esto borra las tablas y gráficos guardados, subiendo solo tu código Python limpio.
*   **La alternativa avanzada:** Recomendada para la industria. Instalaremos una herramienta llamada `nbstripout`, que limpia automáticamente las salidas de los notebooks justo antes de que Git haga el commit, sin que tengas que acordarte de hacerlo manualmente. Para configurarlo, ejecuta en tu terminal:
    ```bash
    pip install nbstripout
    nbstripout --install
    ```