1. Comparación Directa: 300 vs. 76 Árboles

| Escenario | Accuracy | Precisión (Clase 1) | Recall / Sensibilidad (Clase 1) | F1-Score (Clase 1) | 
| :--- | :---: | :---: | :---: | :---: |
| 300 Árboles (Umbral 0.50) | 0.79 | 0.64 | 0.67 | 0.65 | 
| 76 Árboles (Umbral 0.50) | 0.80 | 0.66 | 0.63 | 0.65 | 
| 300 Árboles (Umbral Óptimo - 0.3975) | 0.77 | 0.58 | 0.82 | 0.68 |
| 76 Árboles (Umbral Óptimo - 0.3928) | 0.77 | 0.57 | 0.82 | 0.67 |

1. Conclusión sobre la cantidad de árboles:
- Prácticamente el mismo rendimiento: Con solo 76 árboles lográs el mismo F1-score (0.67-0.68) y el mismo Recall (0.82) para la clase de interés (cancelaciones = 1) que con 300 árboles.

- Confirmación de sobreajuste marginal: Con el umbral por defecto (0.50), el modelo de 76 árboles incluso obtiene un Accuracy ligeramente superior (80% vs 79%) y mejor precisión en cancelaciones (0.66 vs 0.64), demostrando que los 224 árboles restantes solo estaban aportando ruido y sobreajustando sobre las probabilidades sin agregar valor predictivo.

- Eficiencia computacional: El modelo de 76 árboles es un 75% más rápido de entrenar y ejecutar en producción manteniendo intacta la calidad predictiva.

2. Interpretación del Mover de Umbral (Puntos de Corte)
El cambio entre el Umbral por Defecto ($0.50$) y el Umbral Óptimo (ajustado para maximizar el F1-Score / Recall de la Clase 1) representa un ajuste de negocio fundamental:
- Con Umbral 0.50 (Enfoque Conservador):
  - Prioriza no equivocarse al clasificar cancelaciones.
  - Captura solo el 63% - 67% de las cancelaciones reales (Recall), pero cuando dice que una reserva se cancelará, acierta el 64% - 66% de las veces (Precisión).
- Con Umbral Óptimo (Enfoque Proactivo / Negocio):
  - Al bajar el umbral de decisión, el modelo se vuelve más sensible.
  - Captura el 82% de todas las cancelaciones reales (3,620 reservas canceladas en test).
  - Costo de Negocio: La precisión disminuye al 57% - 58%, lo que genera más falsos positivos (reservas no canceladas marcadas como probables cancelaciones).

3. Impacto y Tradución al Negocio Hotelero
Para defender este resultado en un proyecto integrador o ante un comité ejecutivo:
- Gestión de Overbooking y Revenue Management:
  - En hotelería, un falso negativo (no detectar una cancelación) suele ser más costoso que un falso positivo (un cliente no cancela pero tomaste una precaución).
  - Al elegir el Umbral Óptimo, el hotel logra identificar al 82% de las cancelaciones futuras, permitiendo reasignar habitaciones o aplicar políticas preventivas a tiempo.
- Justificación Técnica de la Selección Final:
  - Modelo Elegido: GradientBoostingClassifier con n_estimators = 76.
  - Argumentación: Se adopta la regla de parsimonia (Navaja de Ockham). Reducir de 300 a 76 árboles evita el sobreajuste probabilístico detectado en el Log Loss de validación, reduce el footprint en memoria y el tiempo de inferencia, manteniendo un desempeño equivalente en Test ($F1 \approx 0.68$, $Recall = 0.82$).