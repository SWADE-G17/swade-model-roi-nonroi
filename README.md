# SWADE: Model ROI & Non-ROI Classification

Este repositorio contiene los módulos de procesamiento, entrenamiento y evaluación de modelos de aprendizaje profundo del proyecto SWADE, enfocado en la clasificación de imágenes de resonancia magnética para el soporte en el diagnóstico de la enfermedad de Alzheimer. 

El núcleo de este componente se centra en procesar y evaluar el rendimiento de arquitecturas neuronales utilizando enfoques basados en Regiones de Interés (ROI) y Non-ROI.

## Características principales

* **Procesamiento de neuroimágenes:** Implementación de flujos de trabajo para la carga, normalización y manipulación de volúmenes médicos en formatos estándar de la industria.
* **Segmentación ROI vs. Non-ROI:** Extracción y aislamiento de estructuras cerebrales clave para el análisis del comportamiento del modelo en zonas específicas frente al volumen cerebral completo.
* **Modelos de aprendizaje profundo:** Estructuración de arquitecturas neuronales avanzadas adaptadas para trabajar con tensores tridimensionales y datos complejos.
* **Entrenamiento y optimización:** Funciones dedicadas al ciclo de vida del modelo, incluyendo la gestión de hiperparámetros, almacenamiento de checkpoints y optimización de recursos mediante aceleración por hardware.
