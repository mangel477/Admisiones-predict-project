# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
