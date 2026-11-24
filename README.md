# Dashboard-NLP

Tablero interactivo de Natural Language Processing (NLP) que contrasta la imagen
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

# Uso
```bash
streamlit run tablero.py -- comentarios.csv fuente dmo es,en "Imagen proyectada vs Imagen percibida" --col-texto comentario --max-por-fuente 20000
```

## Significado de cada parámetro

| Parámetro | Significado |
|---|---|
| `streamlit run tablero.py` | Abre el tablero interactivo `tablero.py` en el navegador con Streamlit. |
| `--` | Separa las opciones de Streamlit de los argumentos del script. Todo lo que va después se le pasa a `tablero.py`. |
| `comentarios.csv` | Archivo con los comentarios a analizar (CSV, Excel, JSON o texto), puede ser más de uno. |
| `fuente` | Nombre de la columna que indica el origen de cada comentario (p. ej. TripAdvisor, Instagram, DMO). |
| `dmo` | Valor de la columna de fuente que corresponde a la DMO (imagen proyectada). Las demás fuentes se consideran imagen percibida. |
| `es,en` | Idiomas de los comentarios, separados por coma (español e inglés). |
| `"Imagen proyectada vs Imagen percibida"` | Título que se muestra en el tablero. Va entre comillas porque contiene espacios. |
| `--col-texto comentario` | Nombre de la columna que contiene el texto de los comentarios. |
| `--max-por-fuente 20000` | Máximo de comentarios que se toman por fuente. Si una fuente tiene más, se muestrea al azar. |

## Paleta de colores
Se elige con `--paleta` y controla los **mapas de calor** (matrices de confusión, similitud entre fuentes, prevalencia de tópicos).

```bash
streamlit run tablero.py -- comentarios.csv fuente dmo es,en "Imagen proyectada vs Imagen percibida" --col-texto comentario --max-por-fuente 20000 --paleta viridis
```

| Opción | Descripción |
|---|---|
| `cividis` | Valor por defecto. Pensada para personas con daltonismo. |
| `viridis` | Del morado al amarillo, perceptualmente uniforme. |
| `plasma` | Del azul al amarillo, con tonos magenta. |
| `inferno` | Del negro al amarillo, pasando por rojo. |
| `magma` | Del negro al blanco, pasando por morado y rosa. |
Los colores de los grupos son fijos (paleta **Okabe-Ito**, apta para daltonismo) y además se distinguen por trama o forma, no solo por color.

## Contacto
- ar402482@gmail.com
  
