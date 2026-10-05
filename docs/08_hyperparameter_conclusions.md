# Conclusiones del Ajuste de Hiperparámetros

En esta etapa del proyecto, se realizó el ajuste de hiperparámetros (Hyperparameter Tuning) para los modelos **Random Forest** y **Gradient Boosting** utilizando dos estrategias de validación distintas: **Validación Estática** y **Validación Cruzada Temporal (TimeSeriesSplit)**.

A continuación, se detallan los resultados, la comparativa y la decisión final sobre qué modelo utilizar.

---

## 1. Comparativa de Resultados

### Validación Estática (PredefinedSplit)

En este enfoque, se utilizó un único corte fijo en el tiempo para validar los hiperparámetros. Los resultados suelen ser más optimistas.

| Modelo                | ROC AUC (en Validación) | Accuracy | Precision (Clase 1) | Recall (Clase 1) | F1-Score (Clase 1) |
| :-------------------- | :---------------------: | :------: | :-----------------: | :--------------: | :----------------: |
| **Random Forest**     |         0.9217          |  0.8369  |        0.75         |       0.67       |        0.70        |
| **Gradient Boosting** |         0.9036          |  0.8269  |        0.72         |       0.67       |        0.69        |

### Validación Cruzada Temporal (TimeSeriesSplit)

En este enfoque, el modelo se evaluó a través de 5 cortes temporales distintos, obligándolo a predecir el futuro basándose únicamente en el pasado. Es una métrica mucho más realista y estricta.

| Modelo                | ROC AUC Promedio (CV) | Desviación Estándar (Estabilidad) | Accuracy | Precision (Clase 1) | Recall (Clase 1) | F1-Score (Clase 1) |
| :-------------------- | :-------------------: | :-------------------------------: | :------: | :-----------------: | :--------------: | :----------------: |
| **Random Forest**     |      **0.8479**       |            **0.0271**             |  0.8095  |       0.7262        |      0.5358      |       0.6110       |
| **Gradient Boosting** |        0.8427         |              0.0385               |  0.8068  |       0.6725        |      0.6186      |       0.6403       |

---

## 2. Conclusiones de la Validación Cruzada Temporal

1. **El Ganador Absoluto es Random Forest:**
   - No solo tiene un **AUC promedio ligeramente superior** (0.8479 vs 0.8427).
   - Sino que, mucho más importante en series de tiempo, **es más estable**. Su desviación estándar es menor (0.0271 vs 0.0385). Esto significa que el rendimiento del Random Forest varía menos de un mes a otro; es más robusto ante los cambios de comportamiento a lo largo del tiempo.

2. **La Realidad del CV vs Estático:**
   - En la validación estática, el AUC daba por encima de 0.90. Sin embargo, en la CV temporal, el AUC baja a ~0.84.
   - **Esta es la verdadera prueba de fuego**. La validación cruzada temporal es mucho más estricta porque respeta la flecha del tiempo. Un AUC de 0.84 en un escenario temporal estricto sigue siendo un modelo **muy bueno y útil** para predecir cancelaciones, pero es un número mucho más realista y confiable que el 0.92 estático (que probablemente sufría de cierto sobreajuste a ese corte específico).

---

## 3. ¿Cuál deberías elegir como tu "Modelo Final"?

En el 99% de los casos de negocio donde los datos tienen un componente de tiempo (como reservas de hotel, donde la estacionalidad y las tendencias cambian constantemente), **el modelo ganador debe ser el que se optimizó y evaluó usando Validación Cruzada Temporal (CV)**.

**¿Por qué descartamos el estático?**
La validación estática (hacer un solo split o un solo corte en el tiempo) es muy propensa a dar una falsa sensación de seguridad. El modelo puede haber tenido "suerte" con ese corte específico de datos y el buscador de hiperparámetros tiende a elegir configuraciones más agresivas que memorizan ese conjunto.

Por el contrario, el modelo de CV temporal demostró que **es capaz de predecir el futuro de manera consistente** a través de múltiples cortes de tiempo, eligiendo hiperparámetros más conservadores que generalizan mejor. Es un modelo mucho más robusto y confiable para poner en producción.

---

## 4. Comparativa de Hiperparámetros Elegidos

Es interesante observar cómo la estrategia de validación influye en los hiperparámetros seleccionados por el algoritmo de búsqueda.

_(Nota: Rellenar con las salidas de `best_params_` de cada notebook)_

### Modelos con Validación Estática

- **Random Forest:**
  - `classifier__n_estimators`: 300
  - `classifier__max_depth`: 20
  - `classifier__min_samples_split`: 10
  - `classifier__min_samples_leaf`: 1

- **Gradient Boosting:**
  - `classifier__n_estimators`: 300
  - `classifier__learning_rate`: 0.05
  - `classifier__max_depth`: 7
  - `classifier__min_samples_split`: 10
  - `classifier__min_samples_leaf`: 2

### Modelos con Validación Cruzada Temporal (CV)

- **Random Forest:** Dado que este es nuestro modelo ganador, los hiperparámetros definitivos que se guardarán para producción son:
  - `classifier__n_estimators`: 100
  - `classifier__max_depth`: 30
  - `classifier__min_samples_split`: 2
  - `classifier__min_samples_leaf`: 4

- **Gradient Boosting:**
  - `classifier__n_estimators`: 300 --> análisis early stopping: 76
  - `classifier__learning_rate`: 0.1
  - `classifier__max_depth`: 5
  - `classifier__min_samples_split`: 2
  - `classifier__min_samples_leaf`: 4
