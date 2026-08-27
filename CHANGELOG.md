# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-08-26

### Added

- Una demostración web del modelo, construida con Streamlit, donde se ingresan los datos de un candidato y se obtiene su probabilidad estimada de admisión. La interfaz advierte cuando los datos quedan fuera del rango con el que el modelo fue entrenado, acota las probabilidades imposibles que puede devolver una regresión lineal, señala las estimaciones bajas como poco fiables por la sobreestimación detectada en el análisis de interpretación, y desglosa la contribución de cada atributo a la predicción.
- Carga masiva en la demostración: se sube un archivo CSV con un candidato por fila y se obtienen todas las predicciones de una vez, con plantilla de ejemplo descargable, resumen de cuántas filas quedan marcadas por cada advertencia y descarga de resultados. Los nombres de columna se reconocen sin distinguir mayúsculas ni espacios.
- Instrucciones de ejecución de la demostración en el README.
- Las dependencias de ejecución del proyecto quedan declaradas en `pyproject.toml`: hasta ahora estaban instaladas a mano y quien clonara el repositorio no podía ejecutar la aplicación.

### Fixed

- Las herramientas necesarias para ejecutar los notebooks quedan declaradas como dependencias de desarrollo. Faltaban Jupyter, matplotlib y seaborn, que estaban instalados a mano en el entorno local: quien clonara el repositorio y sincronizara el entorno no podía abrir ni ejecutar ningún notebook, y quien ya los tuviera instalados los perdía al sincronizar, porque el comando elimina lo que no está declarado.

## [0.4.0] - 2026-08-26

### Added

- Un notebook de interpretación del modelo que carga el artefacto entregado sin reentrenarlo, identifica los atributos más influyentes mediante coeficientes e importancia por permutación, y caracteriza los errores que comete. El diagnóstico muestra que el error se concentra en el quintil de menor probabilidad real, donde el modelo sobreestima de forma sistemática, y descarta con evidencia que la causa sean valores atípicos, errores de ingreso, la codificación, la regularización o la familia del modelo.
- Un listado de diez pruebas y experimentos para la siguiente iteración, derivados del diagnóstico de errores.

### Added

- Un notebook de selección de modelos que evalúa ocho familias de regresión sobre el mismo pipeline de preparación, descarta las que rinden por debajo del promedio, compara las restantes con validación cruzada y una prueba estadística que corrige el solapamiento entre particiones, y optimiza los hiperparámetros de las tres mejores. El modelo seleccionado, una regresión Ridge, reduce el error medio de predicción un 32% respecto del modelo base.
- El pipeline de preprocesamiento y modelo entrenado se almacena en `models/modelo-seleccion-admisiones.joblib`, con verificación de que el artefacto recargado reproduce las mismas predicciones.

### Fixed

- La deduplicación no detectaba registros repetidos que solo diferían en la posición de un valor faltante, porque un nulo nunca se considera igual a otro. Al imputar, esas filas quedaban idénticas y un mismo registro podía repartirse entre entrenamiento y prueba, filtrando información hacia la evaluación. Ahora se comparan las filas por compatibilidad en las columnas observadas: el conjunto pasa de 471 a 400 registros distintos, desaparecen los valores faltantes —estaban íntegramente en las copias— y la verificación posterior a la transformación no encuentra duplicados ni solapamiento entre particiones.

### Changed

- Los notebooks de ingeniería de atributos y de modelo base recalculan sus resultados sobre los 400 registros distintos: la partición pasa a 320 filas de entrenamiento y 80 de prueba, y las métricas del modelo base ganan estabilidad, con la desviación del R² entre particiones reducida a la mitad.

### Added

- Un notebook de modelo base que establece la referencia contra la que se compararán los modelos de machine learning: reutiliza el pipeline de preparación, replica la misma partición de entrenamiento y prueba, e implementa `ModeloBase`, una heurística sobre `CGPA` con la interfaz de estimador de scikit-learn. Se evalúa con MAE, RMSE y R² frente a modelos `Dummy` de media y mediana.
- Auditoría de duplicados posterior a la transformación de datos, que documenta la fuga de información detectada entre las particiones de entrenamiento y prueba y cuantifica los registros repetidos ocultos tras valores faltantes.
- Validación cruzada del modelo base con diez particiones y curvas de aprendizaje, escalabilidad, dispersión entre particiones y distribución del error, con el preprocesamiento encadenado en el pipeline para que se reajuste en cada partición.
- Selección justificada del predictor del modelo base a partir de su asociación con el objetivo, con verificación empírica del desempeño de cada variable candidata.
- Recomendaciones para construir el modelo de machine learning a partir de los resultados del modelo base.

### Fixed

- El codificador de la variable `Research` fallaba al reajustarse sobre subconjuntos sin valores faltantes y transformar después datos que sí los contenían; ahora los trata como desconocidos y los deriva al imputador.

## [0.3.0] - 2026-08-21

### Added

- Datos brutos de admisiones y documentación de soporte.
- Un notebook para entender el problema de predicción de admisiones.
- Un notebook de análisis exploratorio inicial de los datos de admisiones.
- Datos intermedios de admisiones con tipos corregidos en formato Parquet para análisis posterior.
- Un notebook de análisis exploratorio univariante para caracterizar el conjunto de datos de admisiones.
- Un notebook de análisis exploratorio bivariable que documenta asociaciones entre la probabilidad de admisión, sus predictores y `Research`.
- Se amplió el análisis exploratorio multivariable con asociaciones condicionadas, diagnóstico de multicolinealidad, perfiles conjuntos y una baseline heurística transparente para preparar su validación posterior.
- Un notebook de ingeniería de atributos que construye la preparación de datos con `Pipeline` y `ColumnTransformer` de scikit-learn: elimina registros duplicados, imputa valores faltantes, codifica la variable binaria `Research`, estandariza los predictores numéricos y produce la partición estratificada de entrenamiento y prueba lista para el modelado. Documenta además por qué no se aplicaron la discretización, las transformaciones no lineales ni la selección de atributos.

### Changed

- Se ampliaron los controles de calidad del análisis exploratorio univariante con detección de duplicados, validación de dominios y estadísticas descriptivas para respaldar la preparación de los datos.

## [0.2.0] - 2026-08-20

### Added

- Datos brutos de admisiones y documentación de datos de soporte.
- Un notebook para entender el problema de predicción de admisiones.
- Un notebook de exploración inicial de datos de admisiones y un conjunto de datos intermedio en Parquet con tipos corregidos para análisis posteriores.
- Un notebook de análisis exploratorio de datos univariado para caracterizar el conjunto de datos de admisiones.

### [0.1.0] - 2026-08-15

### Added
- Se incorporaron los datos RAW para el proyecto de predicción de admisión a programas de posgrado.
- Se añadió un notebook de entendimiento del problema que documenta el objetivo, supuestos, fuente de datos, estrategia de actualización y métricas iniciales de evaluación.
