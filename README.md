# Dashboard-NLP

Tablero interactivo de procesamiento de lenguaje natural (PLN) que contrasta la imagen
que una organización de gestión de destino (DMO) proyecta en sus comunicaciones con la
imagen que los visitantes expresan en reseñas y publicaciones de otras fuentes
(TripAdvisor, Instagram y Reddit).

## Objetivo 
Identificar coincidencias y brechas entre el discurso oficial del destino y la
experiencia que perciben los usuarios, para apoyar decisiones de comunicación y
gestión turística.

## Enfoque 
El análisis combina técnicas de minería de texto:

- **Limpieza y detección de atípicos:** Isolation Forest sobre representaciones TF-IDF.
- **Análisis de sentimiento:** clasificador híbrido BERT + léxico multilingüe.
- **Modelado de tópicos:** NMF, con el número de tópicos elegido por coherencia NPMI.
- **Comparación semántica:** similitud coseno entre fuentes (TF-IDF y LSA) e índice de alineación. 
- **Precio, valor y costo:** frecuencia y tono de los comentarios relacionados con este concepto.

## Estructura del tablero
1. Comportamiento (atípicos, bigramas y trigramas)
2. Análisis de sentimiento
3. Tópicos
4. Imagen proyectada vs imagen percibida
5. Precio, valor y costo

## Instalación (Mac / Windows)

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso
```bash
streamlit run app.py
```

1. Sube tus archivos en la barra lateral (un archivo con columna de fuente, o un archivo por fuente).
2. Elige la columna de comentarios, la columna de fuente y cuál(es) fuente(s) son la DMO.
3. Pulsa **Ejecutar análisis**. Los mapas t-SNE se calculan aparte (pestaña "Mapas").

Para probar: `datos_ejemplo.csv` (columnas `comentario`, `fuente`; la DMO es `dmo`).

## Estructura

- `app.py`: interfaz Streamlit.
- `analisis.py`: lógica de análisis (sin interfaz; reutilizable desde un script de línea de comandos).
- `requirements.txt`, `datos_ejemplo.csv`.
