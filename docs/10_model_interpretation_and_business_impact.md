# Interpretación del Modelo y Métricas de Impacto de Negocio**, tal como lo tenías en tu plan original.

Ahora que hemos encontrado el **umbral óptimo** (que maximiza el F1-Score sin tocar el Test Set), el siguiente paso lógico es **aplicar ese umbral a nuestro modelo final y evaluar su impacto en el negocio**.

## ¿Qué significa "aplicar el umbral ajustado"?
Normalmente, cuando llamas a `modelo.predict(X)`, Scikit-Learn usa internamente un umbral de `0.5`. Si la probabilidad es $\ge 0.5$, dice "1" (Cancela); si es $< 0.5$, dice "0" (No cancela).

Como nosotros descubrimos que nuestro umbral óptimo es distinto (`0.3961`), ya no podemos usar `modelo.predict(X)`. En su lugar, siempre usaremos `modelo.predict_proba(X)[:, 1]` para obtener las probabilidades, y luego aplicaremos nuestro umbral manualmente con una simple regla lógica: `(probabilidades >= 0.42).astype(int)`.

## Plan para el Notebook 10 (`10_model_interpretation_and_business_impact.ipynb`)

En este nuevo notebook haremos tres cosas clave:

1. **Entrenar el Modelo Definitivo:**
   * Tomaremos nuestro Random Forest ganador.
   * Lo entrenaremos con **TODOS** los datos históricos (`X_train_full`).
   
2. **Interpretación del Modelo (Feature Importance):**
   * Extraeremos la importancia de las variables del Random Forest para responder a la pregunta del negocio: *"¿Cuáles son los factores que más influyen en que un cliente cancele?"* (ej. `lead_time`, `deposit_type`, etc.).
   * Haremos un gráfico de barras claro y profesional.

3. **Impacto de Negocio (La prueba de fuego con el Umbral Ajustado):**
   * Crearemos una función personalizada que reciba las probabilidades y aplique nuestro **umbral ajustado**.
   * Definiremos una **Matriz de Costos/Beneficios** (como te mencioné antes: cuánto cuesta una habitación vacía vs cuánto cuesta una sobreventa).
   * Simularemos tres escenarios en el conjunto de validación (o en el último fold):
     * Escenario A: No usar ningún modelo (no hacer nada).
     * Escenario B: Usar el modelo con el umbral por defecto (0.5).
     * Escenario C: Usar el modelo con nuestro **umbral ajustado**.
   * Mostraremos cuánto dinero (ficticio, pero proporcional) ahorra el Escenario C respecto a los demás.


## Ampliemos el paso n° 1: Entrenar el Modelo Definitivo

Entrenamos con cross validation como antes?? Seguimos sin usar test, y lo dejamos para la atapa final?
La respuesta es: **No, ya no usamos Cross Validation (CV) aquí.** Y sí, **seguimos sin usar el Test Set.**

Te explico exactamente por qué:

### ¿Para qué servía el Cross Validation?
El CV (nuestro `TimeSeriesSplit`) tenía un único propósito: **evaluar y comparar**. Lo usamos en la fase de experimentación para:
1. Saber qué hiperparámetros eran los mejores (Notebook 08b).
2. Saber qué algoritmo era el mejor (Random Forest vs Gradient Boosting).
3. Evaluar el rendimiento promedio sin tocar el Test Set.

Una vez que ya respondimos esas preguntas, el CV ya cumplió su misión. Ya sabemos que nuestro "campeón" es el Random Forest con ciertos hiperparámetros específicos.

### ¿Por qué entrenamos ahora con TODOS los datos históricos (`X_train_full`) de una sola vez?
Piensa en cómo funciona el `TimeSeriesSplit`. En el último fold, el modelo se entrenó usando, digamos, los meses 1 al 10, y se evaluó en los meses 11 y 12. 

Pero si vamos a poner este modelo en producción para predecir el futuro (o si vamos a extraer la importancia de las variables para explicarle el modelo al negocio), **queremos que el modelo haya aprendido de toda la historia disponible, incluyendo los meses 11 y 12**.

Por lo tanto, el procedimiento estándar es:
1. Tomar el algoritmo ganador (Random Forest).
2. Ponerle los hiperparámetros ganadores.
3. Hacerle un `.fit()` simple y directo usando el 100% de `X_train_full` y `y_train_full`. 

Al hacer esto, el modelo aprende de todos los patrones pasados disponibles.

### ¿Y el Test Set?
Exacto, **el Test Set (`X_test_full`) sigue guardado bajo llave**. 

En este Notebook 10, cuando hagamos la simulación del impacto de negocio (cuánto dinero ahorramos), lo haremos simulando sobre el mismo conjunto de entrenamiento (o sobre el último fold si queremos ser muy estrictos), solo para demostrarle al negocio el valor del modelo. 

El Test Set lo usaremos **una única vez** en la última fase, justo antes de guardar el archivo `.pkl` final, para poder decir: *"Señores, este modelo que vamos a poner en producción tiene una precisión comprobada del X% en datos que jamás había visto"*.

¿Queda clara la diferencia entre usar CV para *buscar* el modelo vs entrenar con todo el Train para *construir* el modelo definitivo?

---

La comparación entre la importancia de variables (Feature Importance) de Random Forest (RF) y Gradient Boosting (GB) es una de las partes más reveladoras de tu proyecto integrador. Muestra exactamente las diferencias algorítmicas entre el aprendizaje en paralelo (Bagging) y el aprendizaje secuencial (Boosting).

#### 1.Las 3 Diferencias Principales (RF vs. GB)

**Las variables continuas suben fuertemente en Random Forest**
- adr (Tarifa Media Diaria):
   - Random Forest: $7.45\%$ (5.ª más importante).
   - Gradient Boosting: $1.98\%$ (11.ª posición).
- total_nights (Noches totales):
   - Random Forest: $4.37\%$ (8.ª posición).
   - Gradient Boosting: $1.41\%$ (12.ª posición).

¿Por qué ocurre esto?

Random Forest construye árboles de decisión profundos e independientes a partir de submuestras aleatorias de variables (max_features). Al forzar al algoritmo a elegir subconjuntos aleatorios en cada división, las variables continuas con alta cardinalidad (como adr) tienen muchas oportunidades de ser seleccionadas como puntos de corte (splits), lo que incrementa su importancia basada en impureza de Gini.

**Gradient Boosting concentra su poder en pocos "Features Estrella"**
- En Gradient Boosting, las 4 variables principales (lead_time, market_segment_Online TA, total_of_special_requests, is_portugal) suman casi el $58\%$ de toda la importancia del modelo.
- En Random Forest, esas mismas variables están mucho más repartidas y suavizadas (suman el $41\%$).

¿Por qué ocurre esto?

Gradient Boosting construye árboles de manera secuencial para corregir los errores residenciales del árbol anterior. En las primeras iteraciones, el algoritmo detecta y explota de manera voraz (greedy) las variables con mayor señal global.

**El caso particular de market_segment_Online TA**
- Gradient Boosting: $14.98\%$ (2.ª más importante).
- Random Forest: $5.79\%$ (6.ª posición).
En Gradient Boosting, pertenecer al canal Online TA es una señal crítica para corregir residuos de cancelación, mientras que Random Forest distribuye esa importancia entre otros canales interconectados como market_segment_Offline TA/TO ($2.91\%$) o deposit_type_No Deposit ($2.31\%$).

#### 2.Tabla Comparativa Directa

| Variable | Importancia Gradient Boosting | Importancia Random Forest | Comportamiento del Algoritmo |
| :-------------------- | :-----------------: | :--------------: | :----------------: |
| lead_time | 16.12% | 14.84% | Coincidencia total: es la variable dominante en ambos modelos.
| total_of_special_requests | 14.53% | 10.39% | Altísima importancia en ambos (fuerte indicador de intención). |
| is_portugal | 12.55% | 10.33% | Clave en ambos (efecto del mercado local). |
| market_segment_Online TA | 14.98% | 5.79% | GB la explota masivamente; RF la diluye con otros canales. |
| adr | 1.98% | 7.45% | RF favorece la alta cardinalidad de esta variable continua. |
| total_nights | 1.41% | 4.37% | Mayor peso en RF debido al muestreo aleatorio de características. |

#### 3.¿Cómo defender esta comparación en tu proyecto?
- Consistencia estructural: A pesar de las diferencias mecánicas de cada algoritmo, ambos coinciden en que el núcleo duro de la predicción de cancelaciones reside en:
   - Anticipación del viaje: lead_time.
   - Involucramiento del cliente: total_of_special_requests.
   - Geografía / Mercado: is_portugal.
   - Intermediario / Canal: agent y market_segment_Online TA.

- Justificación de la elección del modelo:
   - Si preferís un modelo que se enfoque intensamente en las señales categóricas y de canal más determinantes con menos árboles, Gradient Boosting es ideal.
   - Si buscás un modelo más suavizado y distribuido que soporte bien la variabilidad de variables numéricas continuas como adr, Random Forest ofrece esa dispersión de riesgo.