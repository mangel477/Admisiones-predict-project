"""Streamlit demo for the graduate admission prediction model.

HOW TO RUN THE APP:
    uv run streamlit run notebooks/7-deploy/streamlit-app.py

The app loads the pipeline selected in the model selection stage
(preprocessing + Ridge regression) and predicts the probability of admission
from the data entered in the form.
"""

from pathlib import Path
from typing import cast

import pandas as pd
import streamlit as st
from joblib import load
from sklearn.pipeline import Pipeline

# https://docs.streamlit.io/library/api-reference

MODEL_FILE = "modelo-seleccion-admisiones.joblib"

# below this predicted value the model is known to overestimate: the lowest
# quintile shows a mean residual of -0.089 and a mean absolute error of 0.118
UMBRAL_CONFIANZA = 0.60
SESGO_QUINTIL_BAJO = 0.089

# the training columns carry trailing spaces that the pipeline expects verbatim
COLUMNA_LOR = "LOR "

# ranges actually observed in the training set: outside them the linear model
# extrapolates and can return values outside [0, 1]
RANGOS_ENTRENAMIENTO: dict[str, tuple[float, float]] = {
    "GRE Score": (290, 340),
    "TOEFL Score": (92, 120),
    "University Rating": (1, 5),
    "SOP": (1.0, 5.0),
    COLUMNA_LOR: (1.0, 5.0),
    "CGPA": (6.80, 9.92),
}


def encontrar_raiz_del_proyecto() -> Path:
    """Return the project root, found by walking up until pyproject.toml appears.

    Keeps the app runnable regardless of the directory Streamlit was started from.
    """
    for candidato in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
        if (candidato / "pyproject.toml").exists():
            return candidato
    message = "No se encontró la raíz del proyecto: falta pyproject.toml"
    raise FileNotFoundError(message)


@st.cache_resource
def cargar_modelo() -> Pipeline:
    """Load the serialized preprocessing and model pipeline.

    The result is cached so the artifact is read once per session instead of on
    every interaction with the form.
    """
    ruta = encontrar_raiz_del_proyecto() / "models" / MODEL_FILE
    if not ruta.exists():
        message = f"No se encontró el modelo en {ruta}"
        raise FileNotFoundError(message)
    return cast(Pipeline, load(ruta))


def obtener_datos_del_usuario() -> pd.DataFrame:
    """Collect the form inputs and build the frame the pipeline expects.

    Column names must match the training data exactly, trailing spaces included:
    the ColumnTransformer selects them by name and would fail otherwise.
    """
    datos: dict[str, float | bool] = {}

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Pruebas estandarizadas**")
        datos["GRE Score"] = st.number_input(
            label="Puntaje GRE",
            min_value=260,
            max_value=340,
            value=316,
            step=1,
            help="Rango del examen: 260 a 340",
        )
        datos["TOEFL Score"] = st.number_input(
            label="Puntaje TOEFL",
            min_value=0,
            max_value=120,
            value=107,
            step=1,
            help="Rango del examen: 0 a 120",
        )
        datos["CGPA"] = st.number_input(
            label="Promedio académico acumulado (CGPA)",
            min_value=0.0,
            max_value=10.0,
            value=8.56,
            step=0.01,
            help="El predictor más influyente del modelo",
        )

    with col_b:
        st.markdown("**Postulación**")
        datos["University Rating"] = st.slider(
            label="Calificación de la universidad de origen",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
        )
        datos["SOP"] = st.slider(
            label="Fuerza de la carta de intención (SOP)",
            min_value=1.0,
            max_value=5.0,
            value=3.5,
            step=0.5,
        )
        datos[COLUMNA_LOR] = st.slider(
            label="Fuerza de las cartas de recomendación (LOR)",
            min_value=1.0,
            max_value=5.0,
            value=3.5,
            step=0.5,
        )
        datos["Research"] = st.checkbox(
            label="Tiene experiencia en investigación",
            value=True,
        )

    return pd.DataFrame([datos])


def detectar_extrapolacion(datos: pd.DataFrame) -> list[str]:
    """Return the attributes whose value falls outside the observed training range.

    A linear model does not refuse to extrapolate: it keeps applying the same slope
    and can return probabilities below 0 or above 1. Detecting it lets the app say
    so instead of showing an impossible figure.
    """
    fuera = []
    for atributo, (minimo, maximo) in RANGOS_ENTRENAMIENTO.items():
        valor = float(datos[atributo].iloc[0])
        if valor < minimo or valor > maximo:
            fuera.append(f"{atributo.strip()} = {valor:g} (observado: {minimo:g} a {maximo:g})")
    return fuera


def mostrar_prediccion(probabilidad: float, extrapolados: list[str]) -> None:
    """Render the predicted probability together with its reliability caveats."""
    # the model is not bounded, so an extreme profile can fall outside [0, 1];
    # a probability outside that interval is meaningless to a user
    acotada = min(max(probabilidad, 0.0), 1.0)
    porcentaje = acotada * 100

    st.metric(label="Probabilidad estimada de admisión", value=f"{porcentaje:.1f} %")
    st.progress(acotada)

    # min/max returns the value unchanged when already inside the range,
    # so an exact comparison is enough to detect that it was clamped
    if probabilidad != acotada:
        st.error(
            f"El modelo devolvió {probabilidad * 100:.1f} %, un valor imposible para una "
            f"probabilidad. Se muestra acotado al rango válido. Ocurre porque una regresión lineal "
            f"no tiene techo ni piso: sigue aplicando la misma pendiente fuera del rango de datos "
            f"con el que fue entrenada."
        )

    if extrapolados:
        st.warning(
            "**Datos fuera del rango de entrenamiento.** El modelo nunca vio valores así, de modo "
            "que está extrapolando y su estimación no es confiable:\n\n- "
            + "\n- ".join(extrapolados)
        )

    if probabilidad < UMBRAL_CONFIANZA:
        estimacion_corregida = max(probabilidad - SESGO_QUINTIL_BAJO, 0.0) * 100
        st.warning(
            f"**Estimación poco fiable.** El análisis de interpretación mostró que el modelo "
            f"sobreestima de forma sistemática en este rango: en el quintil de menor probabilidad "
            f"el valor real es en promedio {SESGO_QUINTIL_BAJO * 100:.1f} puntos inferior al "
            f"predicho, con un error absoluto medio de 11.8 puntos. La probabilidad real podría "
            f"estar cerca del {estimacion_corregida:.1f} %. Conviene contrastar este resultado con "
            f"un criterio humano antes de tomar una decisión."
        )
    else:
        st.info(
            "El modelo se equivoca en promedio 4.7 puntos porcentuales en este rango. "
            "La estimación es orientativa y no sustituye los criterios de admisión del programa."
        )


def mostrar_contribuciones(modelo: Pipeline, datos: pd.DataFrame) -> None:
    """Break the prediction down into each attribute's contribution.

    The model is linear, so the prediction is the intercept plus the sum of
    coefficient times standardized value. That decomposition is exact, not an
    approximation, and explains why the prediction is what it is.
    """
    preprocesador = modelo.named_steps["preprocessor"]
    regresor = modelo.named_steps["model"]

    transformado = preprocesador.transform(datos)
    contribuciones = pd.DataFrame(
        {
            "contribución": regresor.coef_ * transformado.to_numpy()[0],
            "valor estandarizado": transformado.to_numpy()[0],
        },
        index=[nombre.split("__", 1)[1] for nombre in preprocesador.get_feature_names_out()],
    ).sort_values("contribución", key=abs, ascending=False)

    st.caption(
        f"Punto de partida (promedio del conjunto de entrenamiento): "
        f"{regresor.intercept_ * 100:.1f} %. Cada atributo suma o resta sobre esa base."
    )
    st.bar_chart(contribuciones["contribución"])
    st.dataframe(contribuciones.round(4), width="stretch")


def main() -> None:
    """Draw the page and run the prediction when the form is submitted."""
    st.set_page_config(page_title="Predicción de admisión", page_icon="🎓", layout="centered")

    st.title("🎓 Predicción de admisión a posgrado")
    st.markdown(
        "Estima la probabilidad de que un candidato sea admitido en un programa de posgrado, "
        "a partir de sus puntajes y de su perfil de postulación."
    )

    try:
        modelo = cargar_modelo()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    with st.form("formulario_admision"):
        datos = obtener_datos_del_usuario()
        enviado = st.form_submit_button("Calcular probabilidad", width="stretch")

    if enviado:
        probabilidad = float(modelo.predict(datos)[0])

        st.divider()
        mostrar_prediccion(probabilidad, detectar_extrapolacion(datos))

        with st.expander("¿Por qué esta predicción?"):
            mostrar_contribuciones(modelo, datos)

        with st.expander("Datos enviados al modelo"):
            st.dataframe(datos, width="stretch")

    with st.expander("Sobre el modelo y sus límites"):
        st.markdown(
            """
            El modelo es una regresión **Ridge** seleccionada tras comparar ocho familias de
            algoritmos con validación cruzada y pruebas estadísticas. Sobre el conjunto de prueba
            alcanza un error absoluto medio de **4.7 puntos porcentuales** y explica el **75 %** de
            la varianza.

            **Qué sabe.** `CGPA` es con diferencia el predictor más influyente: un punto más
            equivale a unos 10.8 puntos porcentuales más de probabilidad estimada. Le siguen el
            puntaje GRE y la fuerza de las cartas de recomendación.

            **Qué no sabe.** El modelo desconoce el programa solicitado, la institución de
            procedencia, la cohorte y el contenido real de las cartas. Dos candidatos con idénticos
            puntajes pueden tener resultados distintos por factores que no están en los datos.

            **Dónde falla.** El error se concentra en los candidatos con menor probabilidad real,
            a quienes el modelo **sobreestima de forma sistemática**. Por eso las estimaciones por
            debajo del 60 % aparecen marcadas como poco fiables.

            **Rango de datos con el que fue entrenado.** GRE de 290 a 340, TOEFL de 92 a 120 y
            CGPA de 6.80 a 9.92. Fuera de esos rangos el modelo extrapola: sigue aplicando la misma
            pendiente sobre datos que nunca vio, y puede devolver valores imposibles. La app avisa
            cuando eso ocurre.

            Esta demostración es orientativa y no debe usarse como criterio único de decisión.
            """
        )


if __name__ == "__main__":
    main()
